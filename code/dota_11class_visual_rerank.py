# -*- coding: utf-8 -*-
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor
from torchvision import transforms as T


HOME = os.environ["USERPROFILE"]
DEVICE = "cuda"

DATA_ROOT = os.path.join(HOME, "modeling", "dota_data", "DOTA_V1.5", "train")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
LABEL_DIR = os.path.join(
    DATA_ROOT, "labelTxt-v1.5", "DOTA-v1.5_train"
)

GDINO_DIR = os.path.join(
    HOME, ".cache", "huggingface", "hub",
    "models--IDEA-Research--grounding-dino-tiny",
    "snapshots",
    "a2bb814dd30d776dcf7e30523b00659f4f141c71",
)

DINO_DIR = os.path.join(
    HOME, ".cache", "modelscope", "tiansz", "dinov2-base"
)

BANK_PATH = os.path.join(
    HOME, "Desktop", "实验数据", "exemplar_bank_k5_dinov2_layers.pt"
)

OUTPUT_PATH = os.path.join(
    HOME, "Desktop", "dota_11class_visual_rerank_results.json"
)

CACHE_PATH = os.path.join(
    HOME, "Desktop", "dota_11class_visual_rerank_cache.pt"
)

DOTA_TO_SOURCE = {
    "plane": "airplane",
    "ship": "ship",
    "storage-tank": "storage tank",
    "baseball-diamond": "baseball field",
    "tennis-court": "tennis court",
    "basketball-court": "basketball court",
    "ground-track-field": "ground track field",
    "harbor": "harbor",
    "bridge": "bridge",
    "large-vehicle": "vehicle",
    "small-vehicle": "vehicle",
}

PROMPTS = {
    "plane": "airplane",
    "ship": "ship",
    "storage-tank": "storage tank",
    "baseball-diamond": "baseball field",
    "tennis-court": "tennis court",
    "basketball-court": "basketball court",
    "ground-track-field": "ground track field",
    "harbor": "harbor",
    "bridge": "bridge",
    "large-vehicle": "vehicle",
    "small-vehicle": "vehicle",
}

MAX_IMAGES = 20
TILE_SIZE = 768
OVERLAP = 128
DETECT_THRESHOLD = 0.10
MATCH_IOU = 0.25
NMS_IOU = 0.50
NEG_LAMBDA = 0.50
MAX_BOXES_PER_TILE = 100


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def nms(boxes, scores):
    order = np.argsort(np.asarray(scores))[::-1].tolist()
    keep = []

    while order:
        i = order.pop(0)
        keep.append(i)
        order = [
            j for j in order
            if box_iou(boxes[i], boxes[j]) < NMS_IOU
        ]

    return [boxes[i] for i in keep]


def collect_samples(dota_class):
    samples = []

    for filename in sorted(os.listdir(LABEL_DIR)):
        if not filename.endswith(".txt"):
            continue

        image_path = os.path.join(IMAGE_DIR, filename[:-4] + ".png")
        if not os.path.exists(image_path):
            continue

        gt_boxes = []

        with open(
            os.path.join(LABEL_DIR, filename),
            encoding="utf-8"
        ) as handle:
            for line in handle:
                parts = line.split()

                if len(parts) < 9 or parts[8] != dota_class:
                    continue

                quad = list(map(float, parts[:8]))
                xs = quad[0::2]
                ys = quad[1::2]

                gt_boxes.append([
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                ])

        if gt_boxes:
            samples.append((image_path, gt_boxes))

        if len(samples) >= MAX_IMAGES:
            break

    return samples


def tile_origins(width, height):
    step = TILE_SIZE - OVERLAP

    xs = list(range(
        0,
        max(1, width - TILE_SIZE + 1),
        step,
    ))

    ys = list(range(
        0,
        max(1, height - TILE_SIZE + 1),
        step,
    ))

    if not xs or xs[-1] + TILE_SIZE < width:
        xs.append(max(0, width - TILE_SIZE))

    if not ys or ys[-1] + TILE_SIZE < height:
        ys.append(max(0, height - TILE_SIZE))

    return [(x, y) for y in ys for x in xs]


def detect_tiled(image, prompt, processor, model):
    boxes = []
    scores = []

    for offset_x, offset_y in tile_origins(
        image.width,
        image.height,
    ):
        tile = image.crop((
            offset_x,
            offset_y,
            min(offset_x + TILE_SIZE, image.width),
            min(offset_y + TILE_SIZE, image.height),
        ))

        inputs = processor(
            images=tile,
            text=prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            if torch.is_tensor(value)
            else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output = model(**inputs)

        target_size = torch.tensor(
            [tile.size[::-1]],
            device=DEVICE,
        )

        result = processor.post_process_grounded_object_detection(
            output,
            threshold=DETECT_THRESHOLD,
            target_sizes=target_size,
        )[0]

        order = torch.argsort(
            result["scores"],
            descending=True,
        )[:MAX_BOXES_PER_TILE]

        for local_box, score in zip(
            result["boxes"][order].cpu().tolist(),
            result["scores"][order].cpu().tolist(),
        ):
            boxes.append([
                local_box[0] + offset_x,
                local_box[1] + offset_y,
                local_box[2] + offset_x,
                local_box[3] + offset_y,
            ])
            scores.append(float(score))

    keep = nms(boxes, scores)

    def _flatten_keep(v):
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, (list, tuple)):
            out = []
            for item in v:
                out.extend(_flatten_keep(item))
            return out
        return [int(v)]

    keep = [
        i for i in _flatten_keep(keep)
        if 0 <= i < len(boxes)
    ]

    return [
        boxes[i] for i in keep
    ], [
        scores[i] for i in keep
    ]

def encode_crops(image, boxes, dino, dino_pre):
    features = torch.zeros(
        (len(boxes), 768),
        dtype=torch.float32,
    )

    crops = []
    valid_indices = []

    for index, box in enumerate(boxes):
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(image.width, int(box[2]))
        y2 = min(image.height, int(box[3]))

        if x2 <= x1 + 10 or y2 <= y1 + 10:
            continue

        crops.append(image.crop((x1, y1, x2, y2)))
        valid_indices.append(index)

    for start in range(0, len(crops), 64):
        batch_crops = crops[start:start + 64]
        batch_indices = valid_indices[start:start + 64]

        batch = torch.stack([
            dino_pre(crop)
            for crop in batch_crops
        ]).to(DEVICE)

        with torch.no_grad():
            output = dino(batch)
            batch_features = output.hidden_states[11][:, 0, :]
            batch_features = torch.nn.functional.normalize(
                batch_features,
                dim=1,
            ).cpu()

        for index, feature in zip(
            batch_indices,
            batch_features,
        ):
            features[index] = feature

    return features


def build_rows(samples, method_scores):
    rows = []

    for image_index, scores in enumerate(method_scores):
        for box, score in scores:
            rows.append((
                image_index,
                box,
                float(score),
            ))

    return rows


def evaluate(rows, ground_truth):
    rows = sorted(
        rows,
        key=lambda item: item[2],
        reverse=True,
    )

    used = [
        set()
        for _ in ground_truth
    ]

    tp = []
    fp = []

    for image_index, predicted_box, _ in rows:
        candidates = [
            (
                box_iou(predicted_box, gt_box),
                gt_index,
            )
            for gt_index, gt_box in enumerate(
                ground_truth[image_index]
            )
            if gt_index not in used[image_index]
        ]

        best_iou, best_index = max(
            candidates,
            default=(0.0, -1),
        )

        matched = best_iou >= MATCH_IOU

        if matched:
            used[image_index].add(best_index)

        tp.append(1 if matched else 0)
        fp.append(0 if matched else 1)

    if not rows:
        return {
            "ap25": 0.0,
            "best_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": sum(len(g) for g in ground_truth),
        }

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    total_gt = sum(len(g) for g in ground_truth)
    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum)

    ap25 = 0.0

    for target_recall in np.linspace(0, 1, 11):
        valid = recall >= target_recall
        if valid.any():
            ap25 += float(precision[valid].max()) / 11.0

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-12)
    )

    best_index = int(np.argmax(f1))

    return {
        "ap25": float(ap25),
        "best_f1": float(f1[best_index]),
        "best_precision": float(precision[best_index]),
        "best_recall": float(recall[best_index]),
        "tp": int(tp_cum[best_index]),
        "fp": int(fp_cum[best_index]),
        "fn": int(total_gt - tp_cum[best_index]),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")

    os.environ["HF_HUB_OFFLINE"] = "1"

    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    gdino_processor = AutoProcessor.from_pretrained(GDINO_DIR)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_DIR,
        dtype=torch.float32,
    ).eval().to(DEVICE)

    dino_config = AutoConfig.from_pretrained(DINO_DIR)
    dino_config.output_hidden_states = True

    dino_model = AutoModel.from_pretrained(
        DINO_DIR,
        config=dino_config,
    ).eval().to(DEVICE)

    dino_pre = T.Compose([
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
        BANK_PATH,
        map_location="cpu",
    )

    results = {
        "config": {
            "classes": len(DOTA_TO_SOURCE),
            "images_per_class": MAX_IMAGES,
            "tile_size": TILE_SIZE,
            "overlap": OVERLAP,
            "detect_threshold": DETECT_THRESHOLD,
            "match_iou": MATCH_IOU,
            "layer": "DINOv2 L11",
        },
        "classes": {},
    }

    for dota_class, source_class in DOTA_TO_SOURCE.items():
        prompt = PROMPTS[dota_class]
        samples = collect_samples(dota_class)

        ground_truth = [
            gt_boxes
            for _, gt_boxes in samples
        ]

        positive = torch.nn.functional.normalize(
            bank[source_class][10]["prototype"].float(),
            dim=0,
        )

        other_prototypes = torch.stack([
            bank[class_name][10]["prototype"].float()
            for class_name in bank
            if class_name != source_class
        ])

        other_prototypes = torch.nn.functional.normalize(
            other_prototypes,
            dim=1,
        )

        negative = torch.nn.functional.normalize(
            other_prototypes[
                (other_prototypes @ positive).topk(3).indices
            ].mean(dim=0),
            dim=0,
        )

        method_rows = {
            "gdino": [],
            "vep": [],
            "cdvm": [],
        }

        print(
            f"\n[{dota_class}] "
            f"source={source_class}, prompt={prompt}, "
            f"images={len(samples)}",
            flush=True,
        )

        for image_index, (image_path, _) in enumerate(samples):
            image = Image.open(image_path).convert("RGB")

            boxes, gd_scores = detect_tiled(
                image,
                prompt,
                gdino_processor,
                gdino_model,
            )

            features = encode_crops(
                image,
                boxes,
                dino_model,
                dino_pre,
            )

            gd = torch.tensor(gd_scores)
            positive_scores = features @ positive
            negative_scores = features @ negative

            vep_scores = torch.sqrt(
                torch.clamp(gd, 1e-6, 1.0) *
                torch.clamp(positive_scores, 1e-6, 1.0)
            )

            cdvm_scores = torch.sqrt(
                torch.clamp(gd, 1e-6, 1.0) *
                torch.clamp(
                    positive_scores - NEG_LAMBDA * negative_scores,
                    1e-6,
                    1.0,
                )
            )

            for box, score in zip(boxes, gd):
                method_rows["gdino"].append(
                    (image_index, box, float(score))
                )

            for box, score in zip(boxes, vep_scores):
                method_rows["vep"].append(
                    (image_index, box, float(score))
                )

            for box, score in zip(boxes, cdvm_scores):
                method_rows["cdvm"].append(
                    (image_index, box, float(score))
                )

            print(
                f"  {image_index + 1}/{len(samples)} "
                f"boxes={len(boxes)}",
                flush=True,
            )

        class_result = {
            method: evaluate(rows, ground_truth)
            for method, rows in method_rows.items()
        }

        results["classes"][dota_class] = {
            "source_class": source_class,
            "prompt": prompt,
            "metrics": class_result,
        }

        print(
            dota_class,
            json.dumps(class_result, ensure_ascii=False),
            flush=True,
        )

    macro = {}

    for method in ["gdino", "vep", "cdvm"]:
        values = [
            results["classes"][class_name]["metrics"][method]["ap25"]
            for class_name in results["classes"]
        ]
        macro[method] = {
            "macro_ap25": float(np.mean(values)),
            "per_class_ap25": values,
        }

    results["macro"] = macro

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print("\nSaved:", OUTPUT_PATH, flush=True)


if __name__ == "__main__":
    main()






