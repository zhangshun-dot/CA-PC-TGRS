# -*- coding: utf-8 -*-
import argparse
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


USER_DIR = Path(os.environ["USERPROFILE"])
BASE_PATH = (
    USER_DIR / "Desktop" / "实验代码"
    / "dota_11class_visual_rerank.py"
)
OUTPUT_PATH = USER_DIR / "Desktop" / "dota_adaptive_contrast_results.json"

PILOT_CLASSES = [
    "plane",
    "ship",
    "small-vehicle",
]

BETAS = [0.25, 0.5, 1.0]
RING_MARGIN = 0.05
RING_SCALE = 8.0


def load_base():
    spec = importlib.util.spec_from_file_location(
        "dota_base",
        str(BASE_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_nms(base, boxes, scores):
    order = np.argsort(np.asarray(scores))[::-1].tolist()
    keep = []

    while order:
        index = int(order.pop(0))
        keep.append(index)
        order = [
            int(other)
            for other in order
            if base.box_iou(
                boxes[index],
                boxes[int(other)],
            ) < base.NMS_IOU
        ]

    return keep


def encode_crops(crops, dino, preprocess, device):
    features = torch.zeros(
        (len(crops), 768),
        dtype=torch.float32,
    )

    for start in range(0, len(crops), 64):
        batch_crops = crops[start:start + 64]
        batch = torch.stack([
            preprocess(crop)
            for crop in batch_crops
        ]).to(device)

        with torch.no_grad():
            output = dino(batch)
            current = output.hidden_states[11][:, 0, :]
            current = torch.nn.functional.normalize(
                current,
                dim=1,
            ).cpu()

        features[start:start + len(batch_crops)] = current

    return features


def encode_image(image, dino, preprocess, device):
    return encode_crops(
        [image],
        dino,
        preprocess,
        device,
    )[0]


def make_ring_crop(image, box):
    x1, y1, x2, y2 = map(float, box)

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    pad_x = 0.75 * width
    pad_y = 0.75 * height

    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(image.width, int(x2 + pad_x))
    bottom = min(image.height, int(y2 + pad_y))

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    pixels = np.asarray(crop).copy()

    inner_left = max(0, int(x1 - left))
    inner_top = max(0, int(y1 - top))
    inner_right = min(
        pixels.shape[1],
        int(x2 - left),
    )
    inner_bottom = min(
        pixels.shape[0],
        int(y2 - top),
    )

    border = pixels.copy()
    border[
        inner_top:inner_bottom,
        inner_left:inner_right,
    ] = 0

    valid = np.any(border != 0, axis=2)
    if valid.any():
        fill = border[valid].mean(axis=0).astype(np.uint8)
    else:
        fill = np.array([128, 128, 128], dtype=np.uint8)

    pixels[
        inner_top:inner_bottom,
        inner_left:inner_right,
    ] = fill

    return Image.fromarray(pixels)


def append_rows(rows, method, image_index, boxes, scores):
    for box, score in zip(boxes, scores):
        rows[method].append(
            (image_index, box, float(score))
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all 11 DOTA classes.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    base = load_base()
    base.nms = lambda boxes, scores: fixed_nms(
        base,
        boxes,
        scores,
    )

    if args.full:
        classes = list(base.DOTA_TO_SOURCE.keys())
        base.MAX_IMAGES = args.max_images or 20
    else:
        classes = PILOT_CLASSES
        base.MAX_IMAGES = args.max_images or 10

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    os.environ["HF_HUB_OFFLINE"] = "1"
    device = base.DEVICE

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(
        base.GDINO_DIR,
    )
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        base.GDINO_DIR,
        dtype=torch.float32,
    ).eval().to(device)

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

    method_names = [
        "gdino",
        "cdvm_fixed",
        "ring_0p25",
        "ring_0p5",
        "ring_1p0",
    ]

    results = {
        "config": {
            "classes": classes,
            "images_per_class": base.MAX_IMAGES,
            "tile_size": base.TILE_SIZE,
            "overlap": base.OVERLAP,
            "detect_threshold": base.DETECT_THRESHOLD,
            "match_iou": base.MATCH_IOU,
            "nms_fixed": True,
            "ring_margin": RING_MARGIN,
            "ring_scale": RING_SCALE,
            "betas": BETAS,
        },
        "classes": {},
    }

    for dota_class in classes:
        source_class = base.DOTA_TO_SOURCE[dota_class]
        prompt = base.PROMPTS[dota_class]
        samples = base.collect_samples(dota_class)

        ground_truth = [
            gt_boxes
            for _, gt_boxes in samples
        ]

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

        rows = {
            name: []
            for name in method_names
        }

        print(
            f"\n[{dota_class}] "
            f"source={source_class}, "
            f"images={len(samples)}",
            flush=True,
        )

        for image_index, (image_path, _) in enumerate(samples):
            image = Image.open(image_path).convert("RGB")

            boxes, gd_scores = base.detect_tiled(
                image,
                prompt,
                processor,
                detector,
            )

            object_features = base.encode_crops(
                image,
                boxes,
                dino,
                preprocess,
            )

            ring_images = [
                make_ring_crop(image, box)
                for box in boxes
            ]

            ring_features = encode_crops(
                ring_images,
                dino,
                preprocess,
                device,
            )

            gd = torch.tensor(
                gd_scores,
                dtype=torch.float32,
            )

            object_similarity = object_features @ positive
            negative_similarity = object_features @ negative
            ring_similarity = ring_features @ positive

            cdvm_scores = torch.sqrt(
                torch.clamp(gd, 1e-6, 1.0)
                * torch.clamp(
                    object_similarity
                    - base.NEG_LAMBDA * negative_similarity,
                    1e-6,
                    1.0,
                )
            )

            contrast = (
                ring_similarity
                - object_similarity
                - RING_MARGIN
            )

            gate = torch.sigmoid(
                RING_SCALE * contrast
            )

            contrast_excess = torch.clamp(
                contrast,
                min=0.0,
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
                "cdvm_fixed",
                image_index,
                boxes,
                cdvm_scores,
            )

            for beta in BETAS:
                method = (
                    "ring_"
                    + str(beta).replace(".", "p")
                )

                scores = (
                    cdvm_scores
                    - beta * gate * contrast_excess
                )

                append_rows(
                    rows,
                    method,
                    image_index,
                    boxes,
                    scores,
                )

            print(
                f"  {image_index + 1}/"
                f"{len(samples)} "
                f"boxes={len(boxes)}",
                flush=True,
            )

        metrics = {
            method: base.evaluate(
                method_rows,
                ground_truth,
            )
            for method, method_rows in rows.items()
        }

        results["classes"][dota_class] = {
            "source_class": source_class,
            "prompt": prompt,
            "metrics": metrics,
        }

        print(
            json.dumps(metrics, ensure_ascii=False),
            flush=True,
        )

    results["macro"] = {}

    for method in method_names:
        values = [
            results["classes"][name]["metrics"]
            [method]["ap25"]
            for name in classes
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
        "\nSaved:",
        OUTPUT_PATH,
        flush=True,
    )


if __name__ == "__main__":
    main()

