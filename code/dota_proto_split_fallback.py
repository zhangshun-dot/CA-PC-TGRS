# -*- coding: utf-8 -*-
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = os.environ["USERPROFILE"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = ("plane", "ship", "baseball-diamond")
NOVEL = ("large-vehicle", "small-vehicle")
CLASSES = BASE + NOVEL

PROMPTS = {
    "plane": "airplane",
    "ship": "ship",
    "baseball-diamond": "baseball field",
    "large-vehicle": "large vehicle",
    "small-vehicle": "small vehicle",
}

SOURCE = {
    "plane": "airplane",
    "ship": "ship",
    "baseball-diamond": "baseball field",
    "large-vehicle": "vehicle",
    "small-vehicle": "vehicle",
}

MAX_IMAGES = 40
NEG_LAMBDA = 0.50
OUTPUT = os.path.join(
    HOME,
    "Desktop",
    "dota_proto_split_fallback_results.json",
)


def load_base():
    roots = [
        Path(HOME) / "Desktop" / "实验代码",
        Path.cwd(),
    ]

    candidates = []

    for root in roots:
        if root.exists():
            candidates.extend(sorted(root.glob("*.py")))

    required = (
        "def detect_tiled",
        "def encode_crops",
        "def evaluate",
    )

    for path in candidates:
        if path.name == Path(__file__).name:
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        if not all(item in text for item in required):
            continue

        spec = importlib.util.spec_from_file_location(
            "dota_base",
            str(path),
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["dota_base"] = module
        spec.loader.exec_module(module)

        print("Using base:", path, flush=True)
        return module

    raise FileNotFoundError(
        "实验代码目录中没有找到包含 detect_tiled、encode_crops、evaluate 的脚本"
    )


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down = torch.nn.Linear(
            768,
            32,
            bias=False,
        )
        self.up = torch.nn.Linear(
            32,
            768,
            bias=False,
        )
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(5, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, features, geometry):
        delta = self.up(self.down(features))
        gate = self.gate(geometry)
        return F.normalize(
            features + gate * delta,
            dim=1,
        )


def get_prototype(bank, names, fallback):
    for name in names:
        if name not in bank:
            continue

        item = bank[name]
        if not isinstance(item, dict):
            continue

        for layer in (10, "10", 11, "11"):
            layer_item = item.get(layer)

            if (
                isinstance(layer_item, dict)
                and "prototype" in layer_item
            ):
                return F.normalize(
                    layer_item["prototype"].float(),
                    dim=0,
                )

    return fallback


def main():
    base = load_base()

    base.MAX_IMAGES = MAX_IMAGES
    base.PILOT_CLASSES = CLASSES
    base.DOTA_TO_SOURCE.update(SOURCE)
    base.PROMPTS.update(PROMPTS)

    os.environ["HF_HUB_OFFLINE"] = "1"

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
        flush=True,
    )

    processor = base.AutoProcessor.from_pretrained(
        base.GDINO_DIR
    )

    gdino = (
        base.AutoModelForZeroShotObjectDetection
        .from_pretrained(
            base.GDINO_DIR,
            dtype=torch.float32,
        )
        .eval()
        .to(DEVICE)
    )

    config = base.AutoConfig.from_pretrained(
        base.DINO_DIR
    )
    config.output_hidden_states = True

    dino = (
        base.AutoModel.from_pretrained(
            base.DINO_DIR,
            config=config,
        )
        .eval()
        .to(DEVICE)
    )

    preprocess = base.T.Compose([
        base.T.Resize(
            (224, 224),
            interpolation=base.T.InterpolationMode.BICUBIC,
        ),
        base.T.ToTensor(),
        base.T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    bank = torch.load(
        base.BANK_PATH,
        map_location="cpu",
    )

    cache = {}

    for cls in CLASSES:
        samples = base.collect_samples(cls)
        rows = []

        for image_index, (image_path, gts) in enumerate(samples):
            image = base.Image.open(
                image_path
            ).convert("RGB")

            boxes, gd_scores = base.detect_tiled(
                image,
                PROMPTS[cls],
                processor,
                gdino,
            )

            features = base.encode_crops(
                image,
                boxes,
                dino,
                preprocess,
            )

            for box, gd, feature in zip(
                boxes,
                gd_scores,
                features,
            ):
                width = max(
                    1.0,
                    box[2] - box[0],
                )
                height = max(
                    1.0,
                    box[3] - box[1],
                )
                area = width * height / (
                    image.width * image.height
                )

                geometry = [
                    math.log(width / height),
                    math.log(area),
                    box[0] / image.width,
                    box[1] / image.height,
                    float(gd),
                ]

                label = float(
                    max(
                        (
                            base.box_iou(box, gt)
                            for gt in gts
                        ),
                        default=0.0,
                    ) >= base.MATCH_IOU
                )

                rows.append({
                    "image": image_index,
                    "box": box,
                    "gd": float(gd),
                    "feature": feature,
                    "geometry": geometry,
                    "label": label,
                })

            print(
                f"[{cls}] "
                f"{image_index + 1}/{len(samples)} "
                f"boxes={len(boxes)}",
                flush=True,
            )

        cache[cls] = (samples, rows)

    positives = [
        row["feature"]
        for cls in BASE
        for row in cache[cls][1]
        if row["label"] > 0.5
    ]

    fallback = F.normalize(
        torch.stack(positives).mean(dim=0),
        dim=0,
    )

    prototypes = {}

    for cls in CLASSES:
        prototypes[cls] = get_prototype(
            bank,
            [
                cls,
                SOURCE[cls],
                PROMPTS[cls],
                cls.replace("-", " "),
            ],
            fallback,
        )

    vehicle_rows = (
        cache["large-vehicle"][1]
        + cache["small-vehicle"][1]
    )

    if vehicle_rows:
        areas = np.asarray([
            math.exp(row["geometry"][1])
            for row in vehicle_rows
        ])

        small_limit, large_limit = np.quantile(
            areas,
            [0.35, 0.65],
        )

        for cls, mask in (
            (
                "small-vehicle",
                areas <= small_limit,
            ),
            (
                "large-vehicle",
                areas >= large_limit,
            ),
        ):
            selected = [
                row["feature"]
                for row, keep in zip(
                    vehicle_rows,
                    mask,
                )
                if keep and row["gd"] >= 0.10
            ]

            if selected:
                prototypes[cls] = F.normalize(
                    torch.stack(selected).mean(dim=0),
                    dim=0,
                )

    adapter = Adapter().to(DEVICE)

    train_rows = [
        row
        for cls in BASE
        for row in cache[cls][1]
    ]

    if train_rows:
        features = torch.stack([
            row["feature"]
            for row in train_rows
        ]).to(DEVICE)

        geometry = torch.tensor([
            row["geometry"]
            for row in train_rows
        ], device=DEVICE)

        targets = torch.tensor([
            row["label"]
            for row in train_rows
        ], device=DEVICE)

        train_proto = F.normalize(
            torch.stack([
                prototypes[cls]
                for cls in BASE
            ]).mean(dim=0).to(DEVICE),
            dim=0,
        )

        optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=2e-3,
            weight_decay=1e-4,
        )

        adapter.train()

        for _ in range(80):
            adapted = adapter(
                features,
                geometry,
            )

            scores = (
                adapted @ train_proto
            ).clamp(-1.0, 1.0)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                8.0 * scores,
                targets,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    adapter.eval()

    results = {
        "config": {
            "base_classes": BASE,
            "novel_classes": NOVEL,
            "images_per_class": MAX_IMAGES,
            "novel_gt_training": False,
            "vehicle_prototype_split": "area_quantile_35_65",
            "adapter": "low_rank_geometry_conditioned",
        },
        "classes": {},
    }

    for cls in CLASSES:
        samples, rows = cache[cls]

        positive = prototypes[cls].to(DEVICE)

        other = torch.stack([
            prototypes[name]
            for name in CLASSES
            if name != cls
        ]).to(DEVICE)

        negative = F.normalize(
            other[
                (other @ positive).topk(
                    min(3, len(other))
                ).indices
            ].mean(dim=0),
            dim=0,
        )

        method_rows = {
            "cdvm": [],
            "adapt": [],
            "reliability_fallback": [],
        }

        for row in rows:
            feature = row["feature"].to(DEVICE)
            geometry = torch.tensor(
                [row["geometry"]],
                device=DEVICE,
            )

            with torch.no_grad():
                adapted_feature = adapter(
                    feature.unsqueeze(0),
                    geometry,
                )[0]

            gd = torch.tensor(
                row["gd"],
                device=DEVICE,
            )

            positive_score = feature @ positive
            negative_score = feature @ negative

            cdvm = torch.sqrt(
                gd.clamp(1e-6, 1.0)
                * (
                    positive_score
                    - NEG_LAMBDA * negative_score
                ).clamp(1e-6, 1.0)
            )

            adapted_score = adapted_feature @ positive
            margin = (
                adapted_score - positive_score
            ).abs()

            gate = (
                torch.sigmoid(
                    12.0 * (margin - 0.015)
                )
                * torch.sigmoid(
                    8.0
                    * (
                        positive_score
                        - negative_score
                        - 0.02
                    )
                )
            )

            adapted = (
                cdvm
                + gate * (0.5 + adapted_score) / 2.0
            ).clamp(0.0, 1.0)

            fused = cdvm + gate * (
                adapted - cdvm
            )

            method_rows["cdvm"].append(
                (row["image"], row["box"], float(cdvm))
            )
            method_rows["adapt"].append(
                (row["image"], row["box"], float(adapted))
            )
            method_rows["reliability_fallback"].append(
                (row["image"], row["box"], float(fused))
            )

        ground_truth = [
            gts
            for _, gts in samples
        ]

        results["classes"][cls] = {
            "metrics": {
                name: base.evaluate(
                    values,
                    ground_truth,
                )
                for name, values in method_rows.items()
            },
            "prototype": "split_or_bank",
        }

    results["macro"] = {}

    for method in (
        "cdvm",
        "adapt",
        "reliability_fallback",
    ):
        all_values = [
            results["classes"][cls]["metrics"][method]["ap25"]
            for cls in CLASSES
        ]

        base_values = [
            results["classes"][cls]["metrics"][method]["ap25"]
            for cls in BASE
        ]

        novel_values = [
            results["classes"][cls]["metrics"][method]["ap25"]
            for cls in NOVEL
        ]

        results["macro"][method] = {
            "macro_ap25": float(np.mean(all_values)),
            "base_macro_ap25": float(np.mean(base_values)),
            "novel_macro_ap25": float(np.mean(novel_values)),
            "per_class_ap25": all_values,
        }

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("Saved:", OUTPUT, flush=True)


if __name__ == "__main__":
    main()

