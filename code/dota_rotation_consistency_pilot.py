# -*- coding: utf-8 -*-
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)
from torchvision import transforms as T


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTIVE_PATH = SCRIPT_DIR / "dota_adaptive_contrast_pilot.py"
OUTPUT_PATH = (
    Path(os.environ["USERPROFILE"])
    / "Desktop"
    / "dota_rotation_consistency_results.json"
)

CLASSES = ["plane", "ship", "small-vehicle"]
MAX_IMAGES = 10
ALPHAS = [0.25, 0.5, 1.0]
MIN_CROSS_IOU = 0.10


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotate_90(image):
    try:
        return image.transpose(Image.Transpose.ROTATE_90)
    except AttributeError:
        return image.transpose(Image.ROTATE_90)


def map_rotated_box_back(box, original_width, original_height):
    """Map a box from a 90-degree CCW image back to the original."""
    rx1, ry1, rx2, ry2 = map(float, box)

    x1 = original_width - ry2
    y1 = rx1
    x2 = original_width - ry1
    y2 = rx2

    return [
        max(0.0, min(float(original_width), x1)),
        max(0.0, min(float(original_height), y1)),
        max(0.0, min(float(original_width), x2)),
        max(0.0, min(float(original_height), y2)),
    ]


def append_rows(rows, method, image_index, boxes, scores):
    for box, score in zip(boxes, scores):
        rows[method].append(
            (image_index, box, float(score))
        )


def cdvm_scores(base, gd_scores, features, positive, negative):
    gd = torch.tensor(gd_scores, dtype=torch.float32)
    positive_scores = features @ positive
    negative_scores = features @ negative

    cdvm = torch.sqrt(
        torch.clamp(gd, 1e-6, 1.0)
        * torch.clamp(
            positive_scores
            - base.NEG_LAMBDA * negative_scores,
            1e-6,
            1.0,
        )
    )

    return gd, cdvm


def cross_view_support(
    base,
    original_boxes,
    original_features,
    mapped_boxes,
    rotated_features,
):
    geometry = torch.zeros(len(original_boxes))
    geometry_semantic = torch.zeros(len(original_boxes))

    if not mapped_boxes:
        return geometry, geometry_semantic

    for index, box in enumerate(original_boxes):
        ious = torch.tensor([
            base.box_iou(box, candidate)
            for candidate in mapped_boxes
        ])

        match_index = int(torch.argmax(ious))
        best_iou = float(ious[match_index])

        if best_iou < MIN_CROSS_IOU:
            continue

        cosine = torch.dot(
            original_features[index],
            rotated_features[match_index],
        ).clamp(-1.0, 1.0)

        semantic = (cosine + 1.0) / 2.0

        geometry[index] = best_iou
        geometry_semantic[index] = best_iou * semantic

    return geometry, geometry_semantic


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    os.environ["HF_HUB_OFFLINE"] = "1"

    adaptive = load_module(
        ADAPTIVE_PATH,
        "dota_adaptive",
    )
    base = adaptive.load_base()

    # Correct the old base script's NMS return contract.
    base.nms = lambda boxes, scores: adaptive.fixed_nms(
        base,
        boxes,
        scores,
    )
    base.MAX_IMAGES = MAX_IMAGES
    device = base.DEVICE

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(
        "Protocol: original + 90-degree view,",
        len(CLASSES),
        "classes x",
        MAX_IMAGES,
        "images",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(base.GDINO_DIR)

    detector = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(
            base.GDINO_DIR,
            dtype=torch.float32,
        )
        .eval()
        .to(device)
    )

    config = AutoConfig.from_pretrained(base.DINO_DIR)
    config.output_hidden_states = True

    dino = AutoModel.from_pretrained(
        base.DINO_DIR,
        config=config,
    ).eval().to(device)

    preprocess = T.Compose([
        T.Resize(
            (224, 224),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    bank = torch.load(
        base.BANK_PATH,
        map_location="cpu",
    )

    methods = ["gdino", "cdvm"]

    for alpha in ALPHAS:
        suffix = str(alpha).replace(".", "p")
        methods.append("rotation_geo_" + suffix)
        methods.append("rotation_sem_" + suffix)

    results = {
        "config": {
            "classes": CLASSES,
            "images_per_class": MAX_IMAGES,
            "views": ["0", "90_ccw"],
            "alphas": ALPHAS,
            "minimum_cross_view_iou": MIN_CROSS_IOU,
            "nms_fixed": True,
            "formula": (
                "logit(CDVM) + alpha * "
                "cross_view_consistency"
            ),
        },
        "classes": {},
    }

    for class_name in CLASSES:
        source_class = base.DOTA_TO_SOURCE[class_name]
        prompt = base.PROMPTS[class_name]
        samples = base.collect_samples(class_name)
        ground_truth = [item[1] for item in samples]

        positive = torch.nn.functional.normalize(
            bank[source_class][10]["prototype"].float(),
            dim=0,
        )

        other = torch.stack([
            bank[name][10]["prototype"].float()
            for name in bank
            if name != source_class
        ])
        other = torch.nn.functional.normalize(other, dim=1)

        negative = torch.nn.functional.normalize(
            other[
                (other @ positive).topk(3).indices
            ].mean(dim=0),
            dim=0,
        )

        rows = {method: [] for method in methods}

        print(
            f"\n[{class_name}] source={source_class}, "
            f"images={len(samples)}",
            flush=True,
        )

        for image_index, (image_path, _) in enumerate(samples):
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            boxes, gd_scores = base.detect_tiled(
                image,
                prompt,
                processor,
                detector,
            )

            original_features = base.encode_crops(
                image,
                boxes,
                dino,
                preprocess,
            )

            gd, cdvm = cdvm_scores(
                base,
                gd_scores,
                original_features,
                positive,
                negative,
            )

            rotated_image = rotate_90(image)

            rotated_boxes, _ = base.detect_tiled(
                rotated_image,
                prompt,
                processor,
                detector,
            )

            rotated_features = base.encode_crops(
                rotated_image,
                rotated_boxes,
                dino,
                preprocess,
            )

            mapped_boxes = [
                map_rotated_box_back(
                    box,
                    width,
                    height,
                )
                for box in rotated_boxes
            ]

            geometry, geometry_semantic = cross_view_support(
                base,
                boxes,
                original_features,
                mapped_boxes,
                rotated_features,
            )

            cdvm_logit = torch.logit(
                cdvm.clamp(1e-5, 1.0 - 1e-5)
            )

            append_rows(
                rows,
                "gdino",
                image_index,
                boxes,
                gd,
            )
            append_rows(
                rows,
                "cdvm",
                image_index,
                boxes,
                cdvm,
            )

            for alpha in ALPHAS:
                suffix = str(alpha).replace(".", "p")

                append_rows(
                    rows,
                    "rotation_geo_" + suffix,
                    image_index,
                    boxes,
                    cdvm_logit + alpha * geometry,
                )

                append_rows(
                    rows,
                    "rotation_sem_" + suffix,
                    image_index,
                    boxes,
                    cdvm_logit
                    + alpha * geometry_semantic,
                )

            matched = int((geometry > 0).sum())

            print(
                f"  {image_index + 1}/{len(samples)} "
                f"original={len(boxes)} "
                f"rotated={len(rotated_boxes)} "
                f"matched={matched}",
                flush=True,
            )

            torch.cuda.empty_cache()

        metrics = {
            method: base.evaluate(
                method_rows,
                ground_truth,
            )
            for method, method_rows in rows.items()
        }

        results["classes"][class_name] = {
            "source_class": source_class,
            "metrics": metrics,
        }

        print(
            json.dumps(metrics, ensure_ascii=False),
            flush=True,
        )

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                results,
                handle,
                indent=2,
                ensure_ascii=False,
            )

    results["macro"] = {}

    for method in methods:
        values = [
            results["classes"][name]
            ["metrics"][method]["ap25"]
            for name in CLASSES
        ]

        results["macro"][method] = {
            "macro_ap25": float(np.mean(values)),
            "per_class_ap25": values,
        }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nMacro:",
        json.dumps(
            results["macro"],
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("Saved:", OUTPUT_PATH, flush=True)


if __name__ == "__main__":
    main()