# -*- coding: utf-8 -*-
import argparse
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18
from transformers import (
    AutoConfig, AutoModel,
    AutoModelForZeroShotObjectDetection, AutoProcessor,
)

HOME = Path(os.environ["USERPROFILE"])
PRIOR_PATH = HOME / "Desktop" / "实验代码" / "dota_cnn_verifier.py"
OUTPUT = HOME / "Desktop" / "dota_shared_conditional_cnn_results.json"

BASE_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond",
    "basketball-court", "ground-track-field", "harbor",
]
NOVEL_CLASSES = [
    "tennis-court", "bridge", "large-vehicle", "small-vehicle",
]
PILOT_BASE = ["plane", "ship", "baseball-diamond"]
PILOT_NOVEL = ["large-vehicle", "small-vehicle"]

MAX_GT_CROPS = 32
FUSION_WEIGHT = 1.0
EPOCHS = 40
BATCH_SIZE = 512


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_split(samples, class_name, count):
    seed = int(
        hashlib.sha256(class_name.encode()).hexdigest()[:8], 16
    )
    samples = list(samples)
    random.Random(seed).shuffle(samples)
    samples = samples[:count]
    n_train = len(samples) // 2
    n_val = len(samples) // 4
    return (
        samples[:n_train],
        samples[n_train:n_train + n_val],
        samples[n_train + n_val:],
    )


def cap_boxes(boxes):
    if len(boxes) <= MAX_GT_CROPS:
        return boxes
    ids = np.linspace(
        0, len(boxes) - 1, MAX_GT_CROPS, dtype=int
    )
    return [boxes[i] for i in ids]


def geometry(boxes, width, height):
    if not boxes:
        return torch.zeros((0, 3), dtype=torch.float32)

    boxes = torch.tensor(boxes, dtype=torch.float32)
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0) / width
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0) / height
    aspect = w / h.clamp(min=1e-6)

    return torch.stack([
        torch.log(w.clamp(min=1e-6)),
        torch.log(h.clamp(min=1e-6)),
        torch.log(aspect.clamp(min=1e-6)),
    ], dim=1)


class SharedConditionalVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_projection = nn.Linear(512, 256)
        self.prototype_projection = nn.Linear(768, 256)
        self.geometry_projection = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 4 + 32, 256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, cnn, prototype, geom):
        cnn = F.normalize(
            F.gelu(self.cnn_projection(cnn)), dim=1
        )
        prototype = F.normalize(
            F.gelu(self.prototype_projection(prototype)), dim=1
        )
        geom = self.geometry_projection(geom)

        fused = torch.cat([
            cnn,
            prototype,
            cnn * prototype,
            torch.abs(cnn - prototype),
            geom,
        ], dim=1)

        return self.head(fused).squeeze(1)


def focal_loss(logits, labels):
    bce = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    probability = torch.sigmoid(logits)
    pt = probability * labels + (1 - probability) * (1 - labels)
    alpha = torch.where(
        labels > 0.5,
        torch.full_like(labels, 0.75),
        torch.full_like(labels, 0.25),
    )
    return (alpha * (1 - pt).pow(2) * bce).mean()


def predict(model, cnn, prototype, geom, device):
    if len(cnn) == 0:
        return torch.zeros(0)

    outputs = []
    for start in range(0, len(cnn), 2048):
        end = start + 2048
        batch_cnn = cnn[start:end].to(device)
        batch_geom = geom[start:end].to(device)
        batch_proto = prototype.unsqueeze(0).expand(
            len(batch_cnn), -1
        ).to(device)

        with torch.no_grad():
            outputs.append(
                model(
                    batch_cnn,
                    batch_proto,
                    batch_geom,
                ).cpu()
            )

    return torch.cat(outputs)


def extract_records(
    class_name,
    samples,
    supervised_count,
    base,
    prior,
    processor,
    detector,
    dino,
    dino_preprocess,
    cnn_backbone,
    cnn_preprocess,
    positive,
    negative,
    device,
):
    records = []

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

        cnn = prior.encode_cnn(
            image, boxes, cnn_backbone,
            cnn_preprocess, device
        )

        labels = prior.candidate_labels(
            base, boxes, gt_boxes
        )
        geom = geometry(boxes, image.width, image.height)

        if index < supervised_count:
            selected_gt = cap_boxes(gt_boxes)
            gt_cnn = prior.encode_cnn(
                image, selected_gt, cnn_backbone,
                cnn_preprocess, device
            )
            gt_geom = geometry(
                selected_gt, image.width, image.height
            )
        else:
            gt_cnn = torch.zeros((0, 512))
            gt_geom = torch.zeros((0, 3))

        records.append({
            "boxes": boxes,
            "ground_truth": gt_boxes,
            "gdino": gdino,
            "cdvm": cdvm,
            "cnn": cnn,
            "geometry": geom,
            "labels": labels,
            "gt_cnn": gt_cnn,
            "gt_geometry": gt_geom,
        })

        print(
            f"  {index + 1}/{len(samples)} boxes={len(boxes)}",
            flush=True,
        )

    return records


def build_examples(data, prototypes, split_names):
    cnn_parts = []
    proto_parts = []
    geom_parts = []
    label_parts = []
    class_names = list(data)

    for class_index, class_name in enumerate(class_names):
        records = []
        for split_name in split_names:
            records.extend(data[class_name][split_name])

        positive_cnn = []
        positive_geom = []
        negative_cnn = []
        negative_geom = []
        negative_score = []

        for record in records:
            if len(record["gt_cnn"]):
                positive_cnn.append(record["gt_cnn"])
                positive_geom.append(record["gt_geometry"])

            matched = record["labels"] == 1
            if matched.any():
                positive_cnn.append(record["cnn"][matched])
                positive_geom.append(record["geometry"][matched])

            rejected = record["labels"] == 0
            if rejected.any():
                negative_cnn.append(record["cnn"][rejected])
                negative_geom.append(record["geometry"][rejected])
                negative_score.append(record["cdvm"][rejected])

        if not positive_cnn or not negative_cnn:
            print("  skip training class:", class_name, flush=True)
            continue

        positive_cnn = torch.cat(positive_cnn)
        positive_geom = torch.cat(positive_geom)
        negative_cnn = torch.cat(negative_cnn)
        negative_geom = torch.cat(negative_geom)
        negative_score = torch.cat(negative_score)

        order = torch.argsort(negative_score, descending=True)
        limit = min(
            len(order),
            max(100, 5 * len(positive_cnn)),
        )
        order = order[:limit]
        negative_cnn = negative_cnn[order]
        negative_geom = negative_geom[order]

        correct = prototypes[class_name]
        wrong_names = [
            name for name in class_names
            if name != class_name
        ]
        wrong = max(
            wrong_names,
            key=lambda name: float(
                prototypes[name] @ correct
            ),
        )
        wrong = prototypes[wrong]

        cnn_parts.extend([
            positive_cnn,
            negative_cnn,
            positive_cnn,
        ])
        geom_parts.extend([
            positive_geom,
            negative_geom,
            positive_geom,
        ])
        proto_parts.extend([
            correct.unsqueeze(0).expand(len(positive_cnn), -1),
            correct.unsqueeze(0).expand(len(negative_cnn), -1),
            wrong.unsqueeze(0).expand(len(positive_cnn), -1),
        ])
        label_parts.extend([
            torch.ones(len(positive_cnn)),
            torch.zeros(len(negative_cnn)),
            torch.zeros(len(positive_cnn)),
        ])

    return {
        "cnn": torch.cat(cnn_parts),
        "prototype": torch.cat(proto_parts),
        "geometry": torch.cat(geom_parts),
        "label": torch.cat(label_parts),
    }


def evaluate_group(base, model, data, prototypes, device):
    methods = [
        "gdino", "cdvm",
        "shared_cnn", "shared_fusion",
    ]
    per_class = {}

    for class_name, records in data.items():
        rows = {name: [] for name in methods}
        ground_truth = [
            record["ground_truth"]
            for record in records
        ]
        prototype = prototypes[class_name]

        for image_index, record in enumerate(records):
            cnn_logits = predict(
                model,
                record["cnn"],
                prototype,
                record["geometry"],
                device,
            )
            fusion = (
                torch.logit(
                    record["cdvm"].clamp(
                        1e-5, 1.0 - 1e-5
                    )
                )
                + FUSION_WEIGHT * cnn_logits
            )

            score_map = {
                "gdino": record["gdino"],
                "cdvm": record["cdvm"],
                "shared_cnn": cnn_logits,
                "shared_fusion": fusion,
            }

            for method, scores in score_map.items():
                for box, score in zip(record["boxes"], scores):
                    rows[method].append((
                        image_index, box, float(score)
                    ))

        per_class[class_name] = {
            method: base.evaluate(
                rows[method], ground_truth
            )
            for method in methods
        }

    macro = {}
    for method in methods:
        values = [
            per_class[name][method]["ap25"]
            for name in per_class
        ]
        macro[method] = {
            "macro_ap25": float(np.mean(values)),
            "per_class_ap25": values,
        }

    return {
        "per_class": per_class,
        "macro": macro,
    }


def train_model(
    examples,
    device,
    epochs,
    validation_callback=None,
):
    torch.manual_seed(0)
    model = SharedConditionalVerifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )

    best_epoch = epochs
    best_score = -1.0
    best_state = None
    count = len(examples["label"])

    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(epoch)
        order = torch.randperm(count, generator=generator)

        for start in range(0, count, BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]

            logits = model(
                examples["cnn"][ids].to(device),
                examples["prototype"][ids].to(device),
                examples["geometry"][ids].to(device),
            )
            labels = examples["label"][ids].to(device)

            loss = focal_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if validation_callback and (
            epoch % 5 == 0 or epoch == epochs
        ):
            model.eval()
            result = validation_callback(model)
            score = result["macro"]["shared_fusion"][
                "macro_ap25"
            ]
            print(
                f"epoch={epoch} "
                f"val_shared_fusion={score:.6f}",
                flush=True,
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--images", type=int, default=40)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    os.environ["HF_HUB_OFFLINE"] = "1"
    device = "cuda"

    prior = load_module(PRIOR_PATH, "cnn_prior")
    adaptive = prior.load_module(
        prior.ADAPTIVE, "adaptive"
    )
    base = adaptive.load_base()
    base.nms = lambda boxes, scores: adaptive.fixed_nms(
        base, boxes, scores
    )
    base.MAX_IMAGES = max(100, args.images)

    if args.pilot:
        base_classes = PILOT_BASE
        novel_classes = PILOT_NOVEL
        output = OUTPUT.with_name(
            "dota_shared_conditional_cnn_pilot_results.json"
        )
    else:
        base_classes = BASE_CLASSES
        novel_classes = NOVEL_CLASSES
        output = OUTPUT

    all_classes = base_classes + novel_classes

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Base classes:", base_classes, flush=True)
    print("Novel classes:", novel_classes, flush=True)

    processor = AutoProcessor.from_pretrained(base.GDINO_DIR)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        base.GDINO_DIR, dtype=torch.float32
    ).eval().to(device)

    config = AutoConfig.from_pretrained(base.DINO_DIR)
    config.output_hidden_states = True
    dino = AutoModel.from_pretrained(
        base.DINO_DIR, config=config
    ).eval().to(device)

    dino_preprocess = T.Compose([
        T.Resize(
            (224, 224),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    weights = ResNet18_Weights.DEFAULT
    cnn = resnet18(weights=weights)
    cnn_backbone = nn.Sequential(
        *list(cnn.children())[:-1]
    ).eval().to(device)
    cnn_preprocess = weights.transforms()

    bank = torch.load(
        base.BANK_PATH, map_location="cpu"
    )

    prototypes = {}
    negatives = {}

    for class_name in all_classes:
        source = base.DOTA_TO_SOURCE[class_name]
        positive = F.normalize(
            bank[source][10]["prototype"].float(),
            dim=0,
        )
        others = torch.stack([
            bank[name][10]["prototype"].float()
            for name in bank if name != source
        ])
        others = F.normalize(others, dim=1)

        prototypes[class_name] = positive
        negatives[class_name] = F.normalize(
            others[
                (others @ positive).topk(3).indices
            ].mean(0),
            dim=0,
        )

    base_data = {}

    for class_name in base_classes:
        pool = base.collect_samples(class_name)
        train, val, test = deterministic_split(
            pool, class_name, args.images
        )
        combined = train + val + test

        print(
            f"\n[{class_name}] "
            f"train={len(train)}, val={len(val)}, "
            f"test={len(test)}",
            flush=True,
        )

        records = extract_records(
            class_name,
            combined,
            len(train) + len(val),
            base, prior, processor, detector,
            dino, dino_preprocess,
            cnn_backbone, cnn_preprocess,
            prototypes[class_name],
            negatives[class_name],
            device,
        )

        a = len(train)
        b = a + len(val)
        base_data[class_name] = {
            "train": records[:a],
            "val": records[a:b],
            "test": records[b:],
        }

    novel_data = {}

    for class_name in novel_classes:
        pool = base.collect_samples(class_name)
        _, _, test = deterministic_split(
            pool, class_name, args.images
        )

        print(
            f"\n[{class_name}] novel test={len(test)}",
            flush=True,
        )

        novel_data[class_name] = extract_records(
            class_name,
            test,
            0,
            base, prior, processor, detector,
            dino, dino_preprocess,
            cnn_backbone, cnn_preprocess,
            prototypes[class_name],
            negatives[class_name],
            device,
        )

    del detector, dino, cnn_backbone
    torch.cuda.empty_cache()

    train_examples = build_examples(
        base_data, prototypes, ["train"]
    )
    val_data = {
        name: base_data[name]["val"]
        for name in base_classes
    }

    model, best_epoch, best_val = train_model(
        train_examples,
        device,
        EPOCHS,
        validation_callback=lambda current: evaluate_group(
            base, current, val_data, prototypes, device
        ),
    )

    print(
        f"Best validation epoch={best_epoch}, "
        f"AP25={best_val:.6f}",
        flush=True,
    )

    final_examples = build_examples(
        base_data, prototypes, ["train", "val"]
    )
    final_model, _, _ = train_model(
        final_examples,
        device,
        best_epoch,
    )
    final_model.eval()

    base_test = {
        name: base_data[name]["test"]
        for name in base_classes
    }

    base_result = evaluate_group(
        base, final_model, base_test,
        prototypes, device
    )
    novel_result = evaluate_group(
        base, final_model, novel_data,
        prototypes, device
    )

    combined = dict(base_test)
    combined.update(novel_data)
    all_result = evaluate_group(
        base, final_model, combined,
        prototypes, device
    )

    results = {
        "config": {
            "protocol": (
                "shared class-conditioned verifier; "
                "base-class training, novel-class testing"
            ),
            "base_classes": base_classes,
            "novel_classes": novel_classes,
            "images_per_class": args.images,
            "best_epoch": best_epoch,
            "fusion_weight": FUSION_WEIGHT,
            "nms_fixed": True,
        },
        "base_test": base_result,
        "novel_test": novel_result,
        "all_test": all_result,
    }

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            results, handle,
            indent=2, ensure_ascii=False,
        )

    print(
        "\nNovel macro:",
        json.dumps(
            novel_result["macro"],
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()
