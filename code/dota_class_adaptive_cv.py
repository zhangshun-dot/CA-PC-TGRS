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


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTIVE_PATH = SCRIPT_DIR / "dota_adaptive_contrast_pilot.py"
OUTPUT_PATH = (
    Path(os.environ["USERPROFILE"])
    / "Desktop"
    / "dota_class_adaptive_cv_results.json"
)

PILOT_CLASSES = ["plane", "ship", "small-vehicle"]

# Positive beta suppresses misleading context.
# Negative beta rewards supportive context.
BETAS = [
    -4.0, -2.0, -1.0, -0.5, -0.25,
    0.0,
    0.25, 0.5, 1.0, 2.0, 4.0,
]
MARGINS = [0.0, 0.025, 0.05, 0.10, 0.20]
CONTRAST_SCALE = 8.0


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scores_for(record, beta, margin):
    contrast = record["contrast"] - margin
    gate = torch.sigmoid(CONTRAST_SCALE * contrast)
    excess = torch.clamp(contrast, min=0.0)
    return record["cdvm"] - beta * gate * excess


def make_rows(records, indices, method=None, beta=0.0, margin=0.05):
    rows = []

    for new_index, old_index in enumerate(indices):
        record = records[old_index]

        if method == "gdino":
            scores = record["gdino"]
        elif method == "cdvm":
            scores = record["cdvm"]
        else:
            scores = scores_for(record, beta, margin)

        for box, score in zip(record["boxes"], scores):
            rows.append((new_index, box, float(score)))

    return rows


def evaluate_subset(
    base,
    records,
    ground_truth,
    indices,
    method=None,
    beta=0.0,
    margin=0.05,
):
    rows = make_rows(
        records,
        indices,
        method=method,
        beta=beta,
        margin=margin,
    )
    subset_gt = [ground_truth[index] for index in indices]
    return base.evaluate(rows, subset_gt)


def append_original_rows(target, records, indices, beta, margin):
    for image_index in indices:
        record = records[image_index]
        scores = scores_for(record, beta, margin)

        for box, score in zip(record["boxes"], scores):
            target.append((image_index, box, float(score)))


def all_rows(records, method):
    rows = []

    for image_index, record in enumerate(records):
        scores = record[method]

        for box, score in zip(record["boxes"], scores):
            rows.append((image_index, box, float(score)))

    return rows


def select_parameters(base, records, ground_truth, train_indices):
    best = None

    for beta in BETAS:
        for margin in MARGINS:
            metrics = evaluate_subset(
                base,
                records,
                ground_truth,
                train_indices,
                beta=beta,
                margin=margin,
            )

            # Prefer AP25, then F1. Exact ties prefer simpler parameters.
            key = (
                metrics["ap25"],
                metrics["best_f1"],
                -abs(beta),
                -abs(margin - 0.05),
            )

            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "beta": beta,
                    "margin": margin,
                    "calibration_metrics": metrics,
                }

    best.pop("key")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Only run plane, ship and small-vehicle.",
    )
    args = parser.parse_args()

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

    classes = (
        PILOT_CLASSES
        if args.pilot
        else list(base.DOTA_TO_SOURCE.keys())
    )

    # Twenty images are divided into two 10-image folds.
    base.MAX_IMAGES = 20
    device = base.DEVICE

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(
        "Classes:",
        len(classes),
        "images/class:",
        base.MAX_IMAGES,
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(base.GDINO_DIR)
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

    results = {
        "config": {
            "protocol": "two-fold out-of-fold calibration",
            "images_per_class": base.MAX_IMAGES,
            "classes": classes,
            "betas": BETAS,
            "margins": MARGINS,
            "contrast_scale": CONTRAST_SCALE,
            "nms_fixed": True,
            "formula": (
                "cdvm - beta_c * sigmoid(8*(contrast-margin_c))"
                " * relu(contrast-margin_c)"
            ),
        },
        "classes": {},
    }

    for class_name in classes:
        source_class = base.DOTA_TO_SOURCE[class_name]
        prompt = base.PROMPTS[class_name]
        samples = base.collect_samples(class_name)

        if len(samples) < 4:
            raise RuntimeError(
                f"{class_name}: insufficient images: {len(samples)}"
            )

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

        records = []

        print(
            f"\n[{class_name}] source={source_class}, "
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

            ring_crops = [
                adaptive.make_ring_crop(image, box)
                for box in boxes
            ]
            ring_features = adaptive.encode_crops(
                ring_crops,
                dino,
                preprocess,
                device,
            )

            gdino = torch.tensor(
                gd_scores,
                dtype=torch.float32,
            )
            object_similarity = object_features @ positive
            negative_similarity = object_features @ negative
            ring_similarity = ring_features @ positive

            cdvm = torch.sqrt(
                torch.clamp(gdino, 1e-6, 1.0)
                * torch.clamp(
                    object_similarity
                    - base.NEG_LAMBDA * negative_similarity,
                    1e-6,
                    1.0,
                )
            )

            records.append({
                "boxes": boxes,
                "gdino": gdino,
                "cdvm": cdvm,
                "contrast": ring_similarity - object_similarity,
            })

            print(
                f"  {image_index + 1}/{len(samples)} "
                f"boxes={len(boxes)}",
                flush=True,
            )

        even = [
            index
            for index in range(len(records))
            if index % 2 == 0
        ]
        odd = [
            index
            for index in range(len(records))
            if index % 2 == 1
        ]

        folds = [
            {"train": odd, "test": even},
            {"train": even, "test": odd},
        ]

        adaptive_rows = []
        selections = []

        for fold_index, fold in enumerate(folds):
            selected = select_parameters(
                base,
                records,
                ground_truth,
                fold["train"],
            )

            append_original_rows(
                adaptive_rows,
                records,
                fold["test"],
                selected["beta"],
                selected["margin"],
            )

            selections.append({
                "fold": fold_index,
                "beta": selected["beta"],
                "margin": selected["margin"],
                "calibration_ap25": (
                    selected["calibration_metrics"]["ap25"]
                ),
                "calibration_f1": (
                    selected["calibration_metrics"]["best_f1"]
                ),
            })

            print(
                f"  fold={fold_index} "
                f"beta={selected['beta']} "
                f"margin={selected['margin']} "
                f"cal_ap25="
                f"{selected['calibration_metrics']['ap25']:.6f}",
                flush=True,
            )

        metrics = {
            "gdino": base.evaluate(
                all_rows(records, "gdino"),
                ground_truth,
            ),
            "cdvm": base.evaluate(
                all_rows(records, "cdvm"),
                ground_truth,
            ),
            "class_adaptive_cv": base.evaluate(
                adaptive_rows,
                ground_truth,
            ),
        }

        results["classes"][class_name] = {
            "source_class": source_class,
            "selected_parameters": selections,
            "metrics": metrics,
        }

        print(
            json.dumps(metrics, ensure_ascii=False),
            flush=True,
        )

        # Preserve completed classes if a later class fails.
        with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
            json.dump(
                results,
                handle,
                indent=2,
                ensure_ascii=False,
            )

    results["macro"] = {}

    for method in ["gdino", "cdvm", "class_adaptive_cv"]:
        values = [
            results["classes"][name]["metrics"][method]["ap25"]
            for name in classes
        ]
        results["macro"][method] = {
            "macro_ap25": float(np.mean(values)),
            "per_class_ap25": values,
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            results,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nMacro:",
        json.dumps(results["macro"], ensure_ascii=False),
        flush=True,
    )
    print("Saved:", OUTPUT_PATH, flush=True)


if __name__ == "__main__":
    main()