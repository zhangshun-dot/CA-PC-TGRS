# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

import dota_geometry_dual_query_token as zero


USER_DIR = Path(os.environ["USERPROFILE"])
CODE_DIR = USER_DIR / "Desktop" / "实验代码"
PRIOR_PATH = CODE_DIR / "dota_cnn_verifier.py"
OUTPUT_PATH = (
    USER_DIR / "Desktop"
    / "dota_learned_dual_query_results.json"
)

BASE_CLASSES = ["plane", "ship", "baseball-diamond"]
NOVEL_CLASSES = ["large-vehicle", "small-vehicle"]

MAX_GT_CROPS = 32
TRAIN_BATCH = 128
EPOCHS = 40
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def box_geometry(boxes, width, height):
    if not boxes:
        return torch.zeros((0, 6))

    boxes = torch.tensor(boxes, dtype=torch.float32)
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1) / width
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1) / height
    area = w * h
    aspect = w / h.clamp(min=1e-6)
    cx = (boxes[:, 0] + boxes[:, 2]) / (2 * width)
    cy = (boxes[:, 1] + boxes[:, 3]) / (2 * height)

    return torch.stack([
        torch.log(w.clamp(min=1e-6)),
        torch.log(h.clamp(min=1e-6)),
        torch.log(area.clamp(min=1e-6)),
        torch.log(aspect.clamp(min=1e-6)),
        cx - 0.5,
        cy - 0.5,
    ], dim=1)


def cap_boxes(boxes):
    if len(boxes) <= MAX_GT_CROPS:
        return boxes

    ids = np.linspace(
        0, len(boxes) - 1,
        MAX_GT_CROPS,
        dtype=int,
    )
    return [boxes[index] for index in ids]


class LowRankAdapter(nn.Module):
    def __init__(self, dimension=768, rank=16):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        residual = self.up(F.gelu(self.down(value)))
        return F.normalize(value + 0.1 * residual, dim=-1)


class LearnedDualQuery(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = LowRankAdapter()

        self.geometry_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        self.query_encoder = nn.Sequential(
            nn.Linear(768, 64),
            nn.GELU(),
        )

        self.geometry_to_query = nn.Linear(
            64, 768, bias=False
        )
        nn.init.zeros_(self.geometry_to_query.weight)

        self.log_temperature = nn.Parameter(
            torch.tensor(-2.7)
        )

        self.dual_scale = nn.Parameter(torch.tensor(2.0))
        self.dual_bias = nn.Parameter(torch.tensor(0.0))
        self.geometry_scale = nn.Parameter(torch.tensor(2.0))
        self.geometry_bias = nn.Parameter(torch.tensor(0.0))

    def evidence(self, tokens, query):
        temperature = (
            F.softplus(self.log_temperature) + 0.025
        )

        similarity = torch.einsum(
            "btd,bd->bt",
            tokens,
            query,
        )

        weights = torch.softmax(
            similarity / temperature,
            dim=1,
        )

        evidence = (
            weights * similarity
        ).sum(dim=1)

        entropy = -(
            weights
            * torch.log(weights.clamp(min=1e-8))
        ).sum(dim=1)

        entropy = entropy / math.log(
            similarity.shape[1]
        )

        return evidence, entropy.clamp(0, 1)

    def conditioned_query(self, query, geometry_code):
        query_code = self.query_encoder(query)
        interaction = geometry_code * query_code
        delta = self.geometry_to_query(interaction)

        return F.normalize(
            query + 0.1 * torch.tanh(delta),
            dim=1,
        )

    def forward(
        self,
        tokens,
        positive,
        negative,
        geometry,
    ):
        tokens = self.adapter(tokens)
        positive = self.adapter(positive)
        negative = self.adapter(negative)

        positive_evidence, positive_entropy = (
            self.evidence(tokens, positive)
        )
        negative_evidence, _ = self.evidence(
            tokens, negative
        )

        dual_margin = (
            positive_evidence
            - 0.5 * negative_evidence
        )

        geometry_code = self.geometry_encoder(geometry)

        conditioned_positive = self.conditioned_query(
            positive,
            geometry_code,
        )

        conditioned_negative = self.conditioned_query(
            negative,
            geometry_code,
        )

        geometry_positive, geometry_entropy = (
            self.evidence(
                tokens,
                conditioned_positive,
            )
        )

        geometry_negative, _ = self.evidence(
            tokens,
            conditioned_negative,
        )

        geometry_margin = (
            geometry_positive
            - 0.5 * geometry_negative
        )

        dual_logit = (
            F.softplus(self.dual_scale)
            * dual_margin
            + self.dual_bias
        )

        geometry_logit = (
            F.softplus(self.geometry_scale)
            * geometry_margin
            + self.geometry_bias
        )

        confidence = torch.sqrt(
            (1.0 - geometry_entropy).clamp(0, 1)
        )

        agreement = torch.exp(
            -4.0 * torch.abs(
                torch.sigmoid(dual_logit)
                - torch.sigmoid(geometry_logit)
            )
        )

        gate = (confidence * agreement).clamp(0, 1)

        return {
            "dual": dual_logit,
            "geometry": geometry_logit,
            "gate": gate,
            "dual_margin": dual_margin,
            "geometry_margin": geometry_margin,
        }


def extract_records(
    class_name,
    samples,
    supervised_count,
    base,
    prior,
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
        image = Image.open(image_path).convert("RGB")

        boxes, gd_scores = base.detect_tiled(
            image,
            base.PROMPTS[class_name],
            processor,
            detector,
        )

        global_feature, tokens = zero.encode_crops(
            image,
            boxes,
            dino,
            preprocess,
            device,
        )

        gdino = torch.tensor(
            gd_scores,
            dtype=torch.float32,
        )

        positive_score = global_feature @ positive
        negative_score = global_feature @ negative

        cdvm = torch.sqrt(
            gdino.clamp(1e-6, 1)
            * (
                positive_score
                - base.NEG_LAMBDA * negative_score
            ).clamp(1e-6, 1)
        )

        if index < supervised_count:
            labels = prior.candidate_labels(
                base, boxes, gt_boxes
            )

            selected_gt = cap_boxes(gt_boxes)
            _, gt_tokens = zero.encode_crops(
                image,
                selected_gt,
                dino,
                preprocess,
                device,
            )

            gt_geometry = box_geometry(
                selected_gt,
                image.width,
                image.height,
            )
        else:
            labels = torch.full(
                (len(boxes),),
                -1,
                dtype=torch.long,
            )

            gt_tokens = torch.zeros(
                (
                    0,
                    zero.TOKEN_GRID * zero.TOKEN_GRID,
                    768,
                ),
                dtype=torch.float32,
            )
            gt_geometry = torch.zeros((0, 6))

        records.append({
            "boxes": boxes,
            "ground_truth": gt_boxes,
            "gdino": gdino,
            "cdvm": cdvm,
            "tokens": tokens.half(),
            "geometry": box_geometry(
                boxes,
                image.width,
                image.height,
            ),
            "labels": labels,
            "gt_tokens": gt_tokens.half(),
            "gt_geometry": gt_geometry,
        })

        print(
            f"  {index + 1}/{len(samples)} "
            f"boxes={len(boxes)}",
            flush=True,
        )

    return records


def build_examples(
    data,
    prototypes,
    negatives,
    split_names,
):
    token_parts = []
    positive_parts = []
    negative_parts = []
    geometry_parts = []
    label_parts = []

    for class_name in data:
        records = []

        for split_name in split_names:
            records.extend(
                data[class_name][split_name]
            )

        positive_tokens = []
        positive_geometry = []
        rejected_tokens = []
        rejected_geometry = []
        rejected_scores = []

        for record in records:
            if len(record["gt_tokens"]):
                positive_tokens.append(
                    record["gt_tokens"]
                )
                positive_geometry.append(
                    record["gt_geometry"]
                )

            matched = record["labels"] == 1

            if matched.any():
                positive_tokens.append(
                    record["tokens"][matched]
                )
                positive_geometry.append(
                    record["geometry"][matched]
                )

            rejected = record["labels"] == 0

            if rejected.any():
                rejected_tokens.append(
                    record["tokens"][rejected]
                )
                rejected_geometry.append(
                    record["geometry"][rejected]
                )
                rejected_scores.append(
                    record["cdvm"][rejected]
                )

        if not positive_tokens or not rejected_tokens:
            print(
                "skip training class:",
                class_name,
                flush=True,
            )
            continue

        positive_tokens = torch.cat(positive_tokens)
        positive_geometry = torch.cat(positive_geometry)
        rejected_tokens = torch.cat(rejected_tokens)
        rejected_geometry = torch.cat(rejected_geometry)
        rejected_scores = torch.cat(rejected_scores)

        order = torch.argsort(
            rejected_scores,
            descending=True,
        )

        limit = min(
            len(order),
            max(128, 5 * len(positive_tokens)),
        )

        order = order[:limit]
        rejected_tokens = rejected_tokens[order]
        rejected_geometry = rejected_geometry[order]

        correct = prototypes[class_name]
        confusing = negatives[class_name]

        token_parts.extend([
            positive_tokens,
            rejected_tokens,
            positive_tokens,
        ])

        geometry_parts.extend([
            positive_geometry,
            rejected_geometry,
            positive_geometry,
        ])

        positive_parts.extend([
            correct.unsqueeze(0).expand(
                len(positive_tokens), -1
            ),
            correct.unsqueeze(0).expand(
                len(rejected_tokens), -1
            ),
            confusing.unsqueeze(0).expand(
                len(positive_tokens), -1
            ),
        ])

        negative_parts.extend([
            confusing.unsqueeze(0).expand(
                len(positive_tokens), -1
            ),
            confusing.unsqueeze(0).expand(
                len(rejected_tokens), -1
            ),
            correct.unsqueeze(0).expand(
                len(positive_tokens), -1
            ),
        ])

        label_parts.extend([
            torch.ones(len(positive_tokens)),
            torch.zeros(len(rejected_tokens)),
            torch.zeros(len(positive_tokens)),
        ])

    return {
        "tokens": torch.cat(token_parts),
        "positive": torch.cat(positive_parts),
        "negative": torch.cat(negative_parts),
        "geometry": torch.cat(geometry_parts),
        "labels": torch.cat(label_parts),
    }


def focal_loss(logits, labels, positive_weight):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )

    probability = torch.sigmoid(logits)

    correct_probability = torch.where(
        labels > 0.5,
        probability,
        1.0 - probability,
    )

    weight = torch.where(
        labels > 0.5,
        torch.full_like(labels, positive_weight),
        torch.ones_like(labels),
    )

    return (
        weight
        * (1.0 - correct_probability).pow(2)
        * bce
    ).mean()


def predict(
    model,
    record,
    positive,
    negative,
    device,
):
    outputs = {
        "dual": [],
        "geometry": [],
        "gate": [],
    }

    for start in range(0, len(record["tokens"]), 256):
        tokens = record["tokens"][
            start:start + 256
        ].float().to(device)

        geometry = record["geometry"][
            start:start + 256
        ].to(device)

        count = len(tokens)

        with torch.inference_mode():
            current = model(
                tokens,
                positive.unsqueeze(0).expand(
                    count, -1
                ).to(device),
                negative.unsqueeze(0).expand(
                    count, -1
                ).to(device),
                geometry,
            )

        for name in outputs:
            outputs[name].append(
                current[name].cpu()
            )

    if not outputs["dual"]:
        return {
            name: torch.zeros(0)
            for name in outputs
        }

    return {
        name: torch.cat(parts)
        for name, parts in outputs.items()
    }


def evaluate_group(
    base,
    model,
    data,
    prototypes,
    negatives,
    device,
    alpha,
):
    methods = [
        "gdino",
        "cdvm",
        "learned_dual",
        "learned_geometry",
        "learned_fusion",
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
            output = predict(
                model,
                record,
                prototypes[class_name],
                negatives[class_name],
                device,
            )

            fusion = (
                torch.logit(
                    record["cdvm"].clamp(
                        1e-5,
                        1 - 1e-5,
                    )
                )
                + alpha
                * output["gate"]
                * output["geometry"]
            )

            scores = {
                "gdino": record["gdino"],
                "cdvm": record["cdvm"],
                "learned_dual": output["dual"],
                "learned_geometry": output["geometry"],
                "learned_fusion": fusion,
            }

            for method in methods:
                for box, score in zip(
                    record["boxes"],
                    scores[method],
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


def select_alpha(
    base,
    model,
    validation_data,
    prototypes,
    negatives,
    device,
):
    best_alpha = 0.0
    best_score = -1.0
    trials = {}

    for alpha in ALPHAS:
        result = evaluate_group(
            base,
            model,
            validation_data,
            prototypes,
            negatives,
            device,
            alpha,
        )

        score = result["macro"]["learned_fusion"][
            "macro_ap25"
        ]

        trials[str(alpha)] = score

        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha, best_score, trials


def train_model(
    examples,
    device,
    epochs,
    validation_callback=None,
):
    torch.manual_seed(0)
    model = LearnedDualQuery().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-3,
    )

    labels = examples["labels"]
    positives = max(1, int((labels == 1).sum()))
    negatives = max(1, int((labels == 0).sum()))
    positive_weight = negatives / positives

    best = {
        "score": -1.0,
        "epoch": epochs,
        "alpha": 0.0,
        "state": None,
        "trials": {},
    }

    for epoch in range(1, epochs + 1):
        model.train()

        generator = torch.Generator().manual_seed(epoch)
        order = torch.randperm(
            len(labels),
            generator=generator,
        )

        total_loss = 0.0
        batches = 0

        for start in range(
            0,
            len(order),
            TRAIN_BATCH,
        ):
            ids = order[start:start + TRAIN_BATCH]
            current_labels = labels[ids].to(device)

            output = model(
                examples["tokens"][ids].float().to(device),
                examples["positive"][ids].to(device),
                examples["negative"][ids].to(device),
                examples["geometry"][ids].to(device),
            )

            dual_loss = focal_loss(
                output["dual"],
                current_labels,
                positive_weight,
            )

            geometry_loss = focal_loss(
                output["geometry"],
                current_labels,
                positive_weight,
            )

            consistency = F.smooth_l1_loss(
                output["geometry_margin"],
                output["dual_margin"].detach(),
            )

            loss = (
                0.4 * dual_loss
                + 0.6 * geometry_loss
                + 0.02 * consistency
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            total_loss += float(loss.detach())
            batches += 1

        if validation_callback and (
            epoch % 5 == 0 or epoch == epochs
        ):
            model.eval()

            alpha, score, trials = (
                validation_callback(model)
            )

            print(
                f"epoch={epoch} "
                f"loss={total_loss / batches:.6f} "
                f"val={score:.6f} "
                f"alpha={alpha}",
                flush=True,
            )

            if score > best["score"]:
                best["score"] = score
                best["epoch"] = epoch
                best["alpha"] = alpha
                best["trials"] = trials
                best["state"] = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in model.state_dict().items()
                }

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    return model, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=40)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    os.environ["HF_HUB_OFFLINE"] = "1"
    device = "cuda"

    prior = zero.load_module(
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

    all_classes = BASE_CLASSES + NOVEL_CLASSES
    prototypes = {}
    confusing_prototypes = {}

    for class_name in all_classes:
        source = base.DOTA_TO_SOURCE[class_name]

        positive = F.normalize(
            bank[source][10]["prototype"].float(),
            dim=0,
        )

        others = torch.stack([
            bank[name][10]["prototype"].float()
            for name in bank
            if name != source
        ])

        others = F.normalize(others, dim=1)

        nearest = (
            others @ positive
        ).topk(3).indices

        prototypes[class_name] = positive
        confusing_prototypes[class_name] = F.normalize(
            others[nearest].mean(dim=0),
            dim=0,
        )

    base_data = {}

    for class_name in BASE_CLASSES:
        pool = base.collect_samples(class_name)

        train, validation, test = (
            zero.deterministic_split(
                pool,
                class_name,
                args.images,
            )
        )

        combined = train + validation + test

        print(
            f"\n[{class_name}] "
            f"train={len(train)}, "
            f"val={len(validation)}, "
            f"test={len(test)}",
            flush=True,
        )

        records = extract_records(
            class_name,
            combined,
            len(train) + len(validation),
            base,
            prior,
            processor,
            detector,
            dino,
            preprocess,
            prototypes[class_name],
            confusing_prototypes[class_name],
            device,
        )

        train_end = len(train)
        validation_end = train_end + len(validation)

        base_data[class_name] = {
            "train": records[:train_end],
            "val": records[
                train_end:validation_end
            ],
            "test": records[validation_end:],
        }

    novel_data = {}

    for class_name in NOVEL_CLASSES:
        pool = base.collect_samples(class_name)

        _, _, test = zero.deterministic_split(
            pool,
            class_name,
            args.images,
        )

        print(
            f"\n[{class_name}] novel test={len(test)}",
            flush=True,
        )

        novel_data[class_name] = extract_records(
            class_name,
            test,
            0,
            base,
            prior,
            processor,
            detector,
            dino,
            preprocess,
            prototypes[class_name],
            confusing_prototypes[class_name],
            device,
        )

    del detector, dino
    torch.cuda.empty_cache()

    train_examples = build_examples(
        base_data,
        prototypes,
        confusing_prototypes,
        ["train"],
    )

    validation_data = {
        name: base_data[name]["val"]
        for name in BASE_CLASSES
    }

    model, best = train_model(
        train_examples,
        device,
        EPOCHS,
        validation_callback=lambda current: select_alpha(
            base,
            current,
            validation_data,
            prototypes,
            confusing_prototypes,
            device,
        ),
    )

    print(
        f"Best epoch={best['epoch']} "
        f"alpha={best['alpha']} "
        f"validation={best['score']:.6f}",
        flush=True,
    )

    final_examples = build_examples(
        base_data,
        prototypes,
        confusing_prototypes,
        ["train", "val"],
    )

    final_model, _ = train_model(
        final_examples,
        device,
        best["epoch"],
    )

    final_model.eval()

    base_test = {
        name: base_data[name]["test"]
        for name in BASE_CLASSES
    }

    base_result = evaluate_group(
        base,
        final_model,
        base_test,
        prototypes,
        confusing_prototypes,
        device,
        best["alpha"],
    )

    novel_result = evaluate_group(
        base,
        final_model,
        novel_data,
        prototypes,
        confusing_prototypes,
        device,
        best["alpha"],
    )

    combined = dict(base_test)
    combined.update(novel_data)

    all_result = evaluate_group(
        base,
        final_model,
        combined,
        prototypes,
        confusing_prototypes,
        device,
        best["alpha"],
    )

    results = {
        "config": {
            "method": (
                "semantic-preserving low-rank "
                "geometry-conditioned dual-query adapter"
            ),
            "base_classes": BASE_CLASSES,
            "novel_classes": NOVEL_CLASSES,
            "images_per_class": args.images,
            "best_epoch": best["epoch"],
            "best_alpha": best["alpha"],
            "validation_score": best["score"],
            "validation_alpha_trials": best["trials"],
            "alpha_zero_included": True,
            "novel_gt_training": False,
            "nms_fixed": True,
        },
        "base_test": base_result,
        "novel_test": novel_result,
        "all_test": all_result,
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
            novel_result["macro"],
            ensure_ascii=False,
        ),
        flush=True,
    )

    print("Saved:", OUTPUT_PATH, flush=True)


if __name__ == "__main__":
    main()