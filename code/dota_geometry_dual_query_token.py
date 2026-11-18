# -*- coding: utf-8 -*-
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)


USER_DIR = Path(os.environ["USERPROFILE"])
CODE_DIR = USER_DIR / "Desktop" / "实验代码"
PRIOR_PATH = CODE_DIR / "dota_cnn_verifier.py"
OUTPUT_PATH = (
    USER_DIR / "Desktop"
    / "dota_geometry_dual_query_token_pilot_results.json"
)

PILOT_BASE = ["plane", "ship", "baseball-diamond"]
PILOT_NOVEL = ["large-vehicle", "small-vehicle"]

TOKEN_GRID = 4
TOKEN_BATCH = 32
FUSION_WEIGHTS = [1.0, 2.0, 4.0, 8.0, 16.0]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_split(samples, class_name, count):
    seed = int(
        hashlib.sha256(
            class_name.encode("utf-8")
        ).hexdigest()[:8],
        16,
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


def crop_boxes(image, boxes):
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

    return crops


def geometry_temperature(boxes, width, height):
    if not boxes:
        return torch.zeros(0)

    boxes = torch.tensor(boxes, dtype=torch.float32)

    box_width = (
        boxes[:, 2] - boxes[:, 0]
    ).clamp(min=1.0) / width

    box_height = (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=1.0) / height

    area = box_width * box_height
    aspect = box_width / box_height.clamp(min=1e-6)

    temperature = (
        0.035
        + 0.080 * torch.sqrt(area)
        + 0.012 * torch.abs(
            torch.log(aspect.clamp(min=1e-6))
        )
    )

    return temperature.clamp(0.035, 0.12)


def pool_patch_tokens(hidden_state):
    patch_tokens = hidden_state[:, 1:, :]
    side = int(math.sqrt(patch_tokens.shape[1]))

    if side < 1:
        raise RuntimeError("DINOv2 did not return patch tokens")

    patch_tokens = patch_tokens[:, :side * side, :]
    patch_tokens = patch_tokens.reshape(
        len(patch_tokens),
        side,
        side,
        patch_tokens.shape[-1],
    )
    patch_tokens = patch_tokens.permute(0, 3, 1, 2)

    patch_tokens = F.adaptive_avg_pool2d(
        patch_tokens,
        (TOKEN_GRID, TOKEN_GRID),
    )

    patch_tokens = patch_tokens.permute(
        0, 2, 3, 1
    ).reshape(
        len(patch_tokens),
        TOKEN_GRID * TOKEN_GRID,
        patch_tokens.shape[1],
    )

    return F.normalize(
        patch_tokens.float(),
        dim=-1,
    )


def encode_crops(
    image,
    boxes,
    dino,
    preprocess,
    device,
):
    if not boxes:
        return (
            torch.zeros((0, 768)),
            torch.zeros((0, TOKEN_GRID * TOKEN_GRID, 768)),
        )

    crops = crop_boxes(image, boxes)
    global_parts = []
    token_parts = []

    for start in range(0, len(crops), TOKEN_BATCH):
        current_crops = crops[start:start + TOKEN_BATCH]

        batch = torch.stack([
            preprocess(crop)
            for crop in current_crops
        ]).to(device)

        with torch.inference_mode():
            output = dino(batch)
            hidden = output.hidden_states[11]

            global_feature = F.normalize(
                hidden[:, 0, :].float(),
                dim=1,
            )

            patch_feature = pool_patch_tokens(hidden)

        global_parts.append(global_feature.cpu())
        token_parts.append(patch_feature.cpu())

    return (
        torch.cat(global_parts),
        torch.cat(token_parts),
    )


def attention_evidence(similarity, temperature):
    if not torch.is_tensor(temperature):
        temperature = torch.tensor(
            temperature,
            dtype=similarity.dtype,
        )

    if temperature.ndim == 0:
        temperature = temperature.expand(
            len(similarity)
        )

    weights = torch.softmax(
        similarity
        / temperature.unsqueeze(1).clamp(min=1e-4),
        dim=1,
    )

    evidence = (
        weights * similarity
    ).sum(dim=1)

    entropy = -(
        weights
        * torch.log(weights.clamp(min=1e-8))
    ).sum(dim=1)

    normalized_entropy = (
        entropy
        / math.log(similarity.shape[1])
    ).clamp(0.0, 1.0)

    return evidence, normalized_entropy


def build_record(
    image_path,
    gt_boxes,
    class_name,
    base,
    processor,
    detector,
    dino,
    preprocess,
    positive,
    negative,
    device,
):
    image = Image.open(image_path).convert("RGB")

    boxes, gd_scores = base.detect_tiled(
        image,
        base.PROMPTS[class_name],
        processor,
        detector,
    )

    gdino = torch.tensor(
        gd_scores,
        dtype=torch.float32,
    )

    global_feature, patch_tokens = encode_crops(
        image,
        boxes,
        dino,
        preprocess,
        device,
    )

    global_positive = global_feature @ positive
    global_negative = global_feature @ negative

    cdvm = torch.sqrt(
        gdino.clamp(1e-6, 1.0)
        * (
            global_positive
            - base.NEG_LAMBDA * global_negative
        ).clamp(1e-6, 1.0)
    )

    positive_similarity = (
        patch_tokens @ positive
    )
    negative_similarity = (
        patch_tokens @ negative
    )

    positive_fixed, _ = attention_evidence(
        positive_similarity,
        0.07,
    )

    negative_fixed, _ = attention_evidence(
        negative_similarity,
        0.07,
    )

    dual_fixed = (
        positive_fixed
        - base.NEG_LAMBDA * negative_fixed
    )

    positive_temperature = geometry_temperature(
        boxes,
        image.width,
        image.height,
    )

    negative_temperature = (
        positive_temperature * 1.25
    ).clamp(0.04, 0.15)

    positive_geometry, positive_entropy = (
        attention_evidence(
            positive_similarity,
            positive_temperature,
        )
    )

    negative_geometry, _ = attention_evidence(
        negative_similarity,
        negative_temperature,
    )

    geometry_dual = (
        positive_geometry
        - base.NEG_LAMBDA * negative_geometry
    )

    attention_confidence = (
        1.0 - positive_entropy
    ).clamp(0.0, 1.0)

    agreement = torch.exp(
        -8.0 * torch.abs(
            dual_fixed - geometry_dual
        )
    )

    gate = (
        attention_confidence * agreement
    ).clamp(0.0, 1.0)

    return {
        "boxes": boxes,
        "ground_truth": gt_boxes,
        "gdino": gdino,
        "cdvm": cdvm,
        "token_positive": positive_fixed,
        "token_dual": dual_fixed,
        "token_geometry": geometry_dual,
        "gate": gate,
    }


def process_samples(
    samples,
    class_name,
    base,
    processor,
    detector,
    dino,
    preprocess,
    positive,
    negative,
    device,
):
    records = []

    for index, (image_path, gt_boxes) in enumerate(samples):
        record = build_record(
            image_path,
            gt_boxes,
            class_name,
            base,
            processor,
            detector,
            dino,
            preprocess,
            positive,
            negative,
            device,
        )
        records.append(record)

        print(
            f"  {index + 1}/{len(samples)} "
            f"boxes={len(record['boxes'])}",
            flush=True,
        )

    return records


def evaluate_group(base, data, fusion_weight):
    methods = [
        "gdino",
        "cdvm",
        "token_positive",
        "token_dual",
        "token_geometry",
        "token_fusion",
    ]

    per_class = {}

    for class_name, records in data.items():
        rows = {
            method: []
            for method in methods
        }

        ground_truth = [
            record["ground_truth"]
            for record in records
        ]

        for image_index, record in enumerate(records):
            fusion = (
                torch.logit(
                    record["cdvm"].clamp(
                        1e-5,
                        1.0 - 1e-5,
                    )
                )
                + fusion_weight
                * record["gate"]
                * record["token_geometry"]
            )

            score_map = {
                "gdino": record["gdino"],
                "cdvm": record["cdvm"],
                "token_positive": record["token_positive"],
                "token_dual": record["token_dual"],
                "token_geometry": record["token_geometry"],
                "token_fusion": fusion,
            }

            for method, scores in score_map.items():
                for box, score in zip(
                    record["boxes"],
                    scores,
                ):
                    rows[method].append((
                        image_index,
                        box,
                        float(score),
                    ))

        per_class[class_name] = {
            method: base.evaluate(
                rows[method],
                ground_truth,
            )
            for method in methods
        }

    macro = {}

    for method in methods:
        values = [
            per_class[class_name][method]["ap25"]
            for class_name in per_class
        ]

        macro[method] = {
            "macro_ap25": float(np.mean(values)),
            "per_class_ap25": values,
        }

    return {
        "per_class": per_class,
        "macro": macro,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--images", type=int, default=40)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    if not PRIOR_PATH.exists():
        raise FileNotFoundError(PRIOR_PATH)

    os.environ["HF_HUB_OFFLINE"] = "1"
    device = "cuda"

    prior = load_module(
        PRIOR_PATH,
        "cnn_prior",
    )

    adaptive = prior.load_module(
        prior.ADAPTIVE,
        "adaptive",
    )

    base = adaptive.load_base()

    base.nms = lambda boxes, scores: (
        adaptive.fixed_nms(
            base,
            boxes,
            scores,
        )
    )

    base.MAX_IMAGES = max(100, args.images)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )
    print(
        "Protocol: training-free token verifier; "
        "base validation selects fusion weight; "
        "novel GT is test-only",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(
        base.GDINO_DIR
    )

    detector = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(
            base.GDINO_DIR,
            dtype=torch.float32,
        )
        .eval()
        .to(device)
    )

    config = AutoConfig.from_pretrained(
        base.DINO_DIR
    )
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
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    bank = torch.load(
        base.BANK_PATH,
        map_location="cpu",
    )

    classes = PILOT_BASE + PILOT_NOVEL
    prototypes = {}
    negatives = {}

    for class_name in classes:
        source = base.DOTA_TO_SOURCE[class_name]

        positive = F.normalize(
            bank[source][10]["prototype"].float(),
            dim=0,
        )

        other_prototypes = torch.stack([
            bank[name][10]["prototype"].float()
            for name in bank
            if name != source
        ])

        other_prototypes = F.normalize(
            other_prototypes,
            dim=1,
        )

        nearest = (
            other_prototypes @ positive
        ).topk(3).indices

        prototypes[class_name] = positive
        negatives[class_name] = F.normalize(
            other_prototypes[nearest].mean(dim=0),
            dim=0,
        )

    validation_data = {}
    base_test_data = {}

    for class_name in PILOT_BASE:
        pool = base.collect_samples(class_name)

        _, validation, test = deterministic_split(
            pool,
            class_name,
            args.images,
        )

        print(
            f"\n[{class_name}] "
            f"validation={len(validation)}, "
            f"test={len(test)}",
            flush=True,
        )

        print(" validation:", flush=True)
        validation_data[class_name] = process_samples(
            validation,
            class_name,
            base,
            processor,
            detector,
            dino,
            preprocess,
            prototypes[class_name],
            negatives[class_name],
            device,
        )

        print(" test:", flush=True)
        base_test_data[class_name] = process_samples(
            test,
            class_name,
            base,
            processor,
            detector,
            dino,
            preprocess,
            prototypes[class_name],
            negatives[class_name],
            device,
        )

    novel_test_data = {}

    for class_name in PILOT_NOVEL:
        pool = base.collect_samples(class_name)

        _, _, test = deterministic_split(
            pool,
            class_name,
            args.images,
        )

        print(
            f"\n[{class_name}] novel test={len(test)}",
            flush=True,
        )

        novel_test_data[class_name] = process_samples(
            test,
            class_name,
            base,
            processor,
            detector,
            dino,
            preprocess,
            prototypes[class_name],
            negatives[class_name],
            device,
        )

    best_weight = FUSION_WEIGHTS[0]
    best_score = -1.0
    validation_trials = {}

    for weight in FUSION_WEIGHTS:
        result = evaluate_group(
            base,
            validation_data,
            weight,
        )

        score = result["macro"]["token_fusion"][
            "macro_ap25"
        ]

        validation_trials[str(weight)] = score

        print(
            f"fusion_weight={weight} "
            f"validation_macro={score:.6f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_weight = weight

    validation_result = evaluate_group(
        base,
        validation_data,
        best_weight,
    )

    base_test_result = evaluate_group(
        base,
        base_test_data,
        best_weight,
    )

    novel_test_result = evaluate_group(
        base,
        novel_test_data,
        best_weight,
    )

    combined_data = dict(base_test_data)
    combined_data.update(novel_test_data)

    all_test_result = evaluate_group(
        base,
        combined_data,
        best_weight,
    )

    results = {
        "config": {
            "method": (
                "training-free geometry-conditioned "
                "contrastive dual-query DINOv2 token verifier"
            ),
            "base_classes": PILOT_BASE,
            "novel_classes": PILOT_NOVEL,
            "images_per_class": args.images,
            "token_grid": TOKEN_GRID,
            "best_fusion_weight": best_weight,
            "validation_trials": validation_trials,
            "nms_fixed": True,
        },
        "validation": validation_result,
        "base_test": base_test_result,
        "novel_test": novel_test_result,
        "all_test": all_test_result,
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
        "\nNovel macro:",
        json.dumps(
            novel_test_result["macro"],
            ensure_ascii=False,
        ),
        flush=True,
    )

    print(
        "\nSaved:",
        OUTPUT_PATH,
        flush=True,
    )


if __name__ == "__main__":
    main()