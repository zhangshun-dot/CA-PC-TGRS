# -*- coding: utf-8 -*-
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import (
    AutoConfig, AutoModel,
    AutoModelForZeroShotObjectDetection, AutoProcessor,
)
from torchvision import transforms as T


HERE = Path(__file__).resolve().parent
ADAPTIVE = HERE / "dota_adaptive_contrast_pilot.py"
OUT = Path(os.environ["USERPROFILE"]) / "Desktop" / "dota_cnn_verifier_results.json"

CLASSES = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle"]
LAMBDAS = [0.1, 0.25, 0.5, 1.0]
MAX_IMAGES = 20
POS_IOU = 0.25
NEG_IOU = 0.05


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_labels(base, boxes, gt_boxes):
    labels = []
    for box in boxes:
        best = max(
            (base.box_iou(box, gt) for gt in gt_boxes),
            default=0.0,
        )
        if best >= POS_IOU:
            labels.append(1)
        elif best < NEG_IOU:
            labels.append(0)
        else:
            labels.append(-1)
    return torch.tensor(labels, dtype=torch.long)


def encode_cnn(image, boxes, backbone, preprocess, device):
    if not boxes:
        return torch.zeros((0, 512))

    crops = []
    for box in boxes:
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(image.width, int(box[2]))
        y2 = min(image.height, int(box[3]))

        if x2 <= x1 + 2 or y2 <= y1 + 2:
            crops.append(Image.new("RGB", (32, 32)))
        else:
            crops.append(image.crop((x1, y1, x2, y2)))

    features = []
    for start in range(0, len(crops), 128):
        batch = torch.stack([
            preprocess(crop)
            for crop in crops[start:start + 128]
        ]).to(device)

        with torch.no_grad():
            current = backbone(batch).flatten(1)
            current = torch.nn.functional.normalize(current, dim=1)

        features.append(current.cpu())

    return torch.cat(features)


def train_head(records, indices, device):
    features = torch.cat([records[i]["cnn"] for i in indices])
    labels = torch.cat([records[i]["labels"] for i in indices])
    cdvm = torch.cat([records[i]["cdvm"] for i in indices])

    valid = labels >= 0
    features = features[valid]
    labels = labels[valid]
    cdvm = cdvm[valid]

    positive = torch.where(labels == 1)[0]
    negative = torch.where(labels == 0)[0]

    if len(positive) == 0 or len(negative) == 0:
        return None, {
            "positive": int(len(positive)),
            "negative": int(len(negative)),
        }

    # Keep high-scoring false candidates as hard negatives.
    negative = negative[
        torch.argsort(cdvm[negative], descending=True)
    ]
    negative = negative[:max(50, 5 * len(positive))]
    selected = torch.cat([positive, negative])

    x = features[selected].to(device)
    y = labels[selected].float().to(device)

    torch.manual_seed(0)
    head = nn.Linear(512, 1).to(device)

    positive_count = max(1, int((y == 1).sum()))
    negative_count = max(1, int((y == 0).sum()))
    pos_weight = torch.tensor(
        [negative_count / positive_count],
        device=device,
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=0.02,
        weight_decay=1e-3,
    )

    head.train()
    for _ in range(300):
        optimizer.zero_grad()
        logits = head(x).squeeze(1)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    return head.cpu().eval(), {
        "positive": positive_count,
        "negative": negative_count,
    }


def add_rows(rows, method, image_index, boxes, scores):
    for box, score in zip(boxes, scores):
        rows[method].append(
            (image_index, box, float(score))
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    os.environ["HF_HUB_OFFLINE"] = "1"
    device = "cuda"

    adaptive = load_module(ADAPTIVE, "adaptive")
    base = adaptive.load_base()
    base.nms = lambda boxes, scores: adaptive.fixed_nms(
        base, boxes, scores
    )
    base.MAX_IMAGES = MAX_IMAGES

    print("GPU:", torch.cuda.get_device_name(0), flush=True)

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

    dino_preprocess = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    weights = ResNet18_Weights.DEFAULT
    cnn_model = resnet18(weights=weights)
    cnn_backbone = nn.Sequential(
        *list(cnn_model.children())[:-1]
    ).eval().to(device)
    cnn_preprocess = weights.transforms()

    bank = torch.load(base.BANK_PATH, map_location="cpu")

    methods = ["gdino", "cdvm", "cnn"]
    methods += [
        "cnn_fusion_" + str(value).replace(".", "p")
        for value in LAMBDAS
    ]

    results = {
        "config": {
            "classes": CLASSES,
            "images_per_class": MAX_IMAGES,
            "protocol": "two-fold out-of-fold CNN verification",
            "cnn": "ImageNet ResNet18 frozen backbone",
            "positive_iou": POS_IOU,
            "negative_iou": NEG_IOU,
            "lambdas": LAMBDAS,
            "nms_fixed": True,
        },
        "classes": {},
    }

    for class_name in CLASSES:
        source = base.DOTA_TO_SOURCE[class_name]
        samples = base.collect_samples(class_name)
        ground_truth = [item[1] for item in samples]

        positive = torch.nn.functional.normalize(
            bank[source][10]["prototype"].float(),
            dim=0,
        )

        others = torch.stack([
            bank[name][10]["prototype"].float()
            for name in bank if name != source
        ])
        others = torch.nn.functional.normalize(others, dim=1)
        negative = torch.nn.functional.normalize(
            others[(others @ positive).topk(3).indices].mean(0),
            dim=0,
        )

        records = []
        print(f"\n[{class_name}] images={len(samples)}", flush=True)

        for index, (image_path, gt_boxes) in enumerate(samples):
            image = Image.open(image_path).convert("RGB")
            boxes, gd_scores = base.detect_tiled(
                image,
                base.PROMPTS[class_name],
                processor,
                detector,
            )

            dino_features = base.encode_crops(
                image, boxes, dino, dino_preprocess
            )
            gdino = torch.tensor(gd_scores, dtype=torch.float32)
            positive_scores = dino_features @ positive
            negative_scores = dino_features @ negative

            cdvm = torch.sqrt(
                gdino.clamp(1e-6, 1.0)
                * (
                    positive_scores
                    - base.NEG_LAMBDA * negative_scores
                ).clamp(1e-6, 1.0)
            )

            cnn = encode_cnn(
                image,
                boxes,
                cnn_backbone,
                cnn_preprocess,
                device,
            )

            records.append({
                "boxes": boxes,
                "gdino": gdino,
                "cdvm": cdvm,
                "cnn": cnn,
                "labels": candidate_labels(
                    base, boxes, gt_boxes
                ),
            })

            print(
                f"  {index + 1}/{len(samples)} boxes={len(boxes)}",
                flush=True,
            )

        even = list(range(0, len(records), 2))
        odd = list(range(1, len(records), 2))
        folds = [(odd, even), (even, odd)]

        rows = {method: [] for method in methods}
        fold_info = []

        for image_index, record in enumerate(records):
            add_rows(
                rows, "gdino", image_index,
                record["boxes"], record["gdino"]
            )
            add_rows(
                rows, "cdvm", image_index,
                record["boxes"], record["cdvm"]
            )

        for train_indices, test_indices in folds:
            head, stats = train_head(
                records, train_indices, device
            )

            fold_info.append(stats)

            for image_index in test_indices:
                record = records[image_index]

                if head is None:
                    cnn_logits = torch.zeros(
                        len(record["boxes"])
                    )
                else:
                    with torch.no_grad():
                        cnn_logits = head(
                            record["cnn"]
                        ).squeeze(1)

                add_rows(
                    rows, "cnn", image_index,
                    record["boxes"], cnn_logits
                )

                cdvm_logits = torch.logit(
                    record["cdvm"].clamp(
                        1e-5, 1.0 - 1e-5
                    )
                )

                for value in LAMBDAS:
                    method = (
                        "cnn_fusion_"
                        + str(value).replace(".", "p")
                    )
                    scores = (
                        cdvm_logits
                        + value * cnn_logits
                    )
                    add_rows(
                        rows, method, image_index,
                        record["boxes"], scores
                    )

        metrics = {
            method: base.evaluate(
                method_rows, ground_truth
            )
            for method, method_rows in rows.items()
        }

        results["classes"][class_name] = {
            "fold_training": fold_info,
            "metrics": metrics,
        }

        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        with open(OUT, "w", encoding="utf-8") as handle:
            json.dump(
                results, handle,
                indent=2, ensure_ascii=False,
            )

    results["macro"] = {}
    for method in methods:
        values = [
            results["classes"][name]["metrics"][method]["ap25"]
            for name in CLASSES
        ]
        results["macro"][method] = {
            "macro_ap25": float(np.mean(values)),
            "per_class_ap25": values,
        }

    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(
            results, handle,
            indent=2, ensure_ascii=False,
        )

    print(
        "\nMacro:",
        json.dumps(results["macro"], ensure_ascii=False),
        flush=True,
    )
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
