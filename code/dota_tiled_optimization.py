import json, os
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

HOME = os.environ["USERPROFILE"]
IMG = os.path.join(HOME, "modeling", "dota_data", "DOTA_V1.5", "train", "images")
LAB = os.path.join(HOME, "modeling", "dota_data", "DOTA_V1.5", "train",
                   "labelTxt-v1.5", "DOTA-v1.5_train")
MODEL = os.path.join(HOME, ".cache", "huggingface", "hub",
    "models--IDEA-Research--grounding-dino-tiny", "snapshots",
    "a2bb814dd30d776dcf7e30523b00659f4f141c71")
OUT = os.path.join(HOME, "Desktop", "dota_tiled_results.json")

MAP = {"plane": "airplane", "ship": "ship", "small-vehicle": "vehicle"}
N, TILE, OVERLAP, TH = 5, 1024, 128, 0.25
DEVICE = "cuda"

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter > 0 else 0

def nms(boxes, scores):
    order = np.argsort(scores)[::-1].tolist()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < 0.5]
    return [boxes[i] for i in keep]

def collect():
    data = defaultdict(list)
    for fn in sorted(os.listdir(LAB)):
        if not fn.endswith(".txt"):
            continue
        image = os.path.join(IMG, fn[:-4] + ".png")
        if not os.path.exists(image):
            continue
        grouped = defaultdict(list)
        with open(os.path.join(LAB, fn), encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 9 or p[8] not in MAP:
                    continue
                q = list(map(float, p[:8]))
                xs, ys = q[0::2], q[1::2]
                grouped[p[8]].append([min(xs), min(ys), max(xs), max(ys)])
        for cls, boxes in grouped.items():
            if len(data[cls]) < N:
                data[cls].append((image, boxes))
    return data

def infer(image, prompt, processor, model):
    x = processor(images=image, text=prompt, return_tensors="pt")
    x = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in x.items()}
    with torch.no_grad():
        output = model(**x)
    size = torch.tensor([image.size[::-1]], device=DEVICE)
    result = processor.post_process_grounded_object_detection(
        output, threshold=TH, target_sizes=size)[0]
    return result["boxes"].cpu().tolist(), result["scores"].cpu().tolist()

def origins(w, h):
    step = TILE - OVERLAP
    xs = list(range(0, max(1, w-TILE+1), step))
    ys = list(range(0, max(1, h-TILE+1), step))
    if not xs or xs[-1]+TILE < w: xs.append(max(0, w-TILE))
    if not ys or ys[-1]+TILE < h: ys.append(max(0, h-TILE))
    return [(x, y) for y in ys for x in xs]

def detect(image, prompt, tiled, processor, model):
    if not tiled:
        return infer(image, prompt, processor, model)[0]
    boxes, scores = [], []
    for ox, oy in origins(*image.size):
        crop = image.crop((ox, oy, min(ox+TILE, image.width),
                           min(oy+TILE, image.height)))
        local_boxes, local_scores = infer(crop, prompt, processor, model)
        for b, score in zip(local_boxes, local_scores):
            boxes.append([b[0]+ox, b[1]+oy, b[2]+ox, b[3]+oy])
            scores.append(score)
    return nms(boxes, scores)

def evaluate(predictions, ground_truth):
    tp = fp = fn = detected = 0
    for preds, gts in zip(predictions, ground_truth):
        detected += bool(preds)
        used = set()
        for pred in preds:
            choices = [(iou(pred, gt), j) for j, gt in enumerate(gts)
                       if j not in used]
            best, idx = max(choices, default=(0, -1))
            if best >= 0.25:
                used.add(idx); tp += 1
            else:
                fp += 1
        fn += len(gts)-len(used)
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return {"dr": detected/len(ground_truth), "precision": p, "recall": r,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn}

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not found")

os.environ["HF_HUB_OFFLINE"] = "1"
print("GPU:", torch.cuda.get_device_name(0))
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL, dtype=torch.float32).eval().to(DEVICE)

methods = {
    "overhead_full": (False, lambda c: f"an overhead image of {c}"),
    "overhead_tiled": (True, lambda c: f"an overhead image of {c}"),
    "tdm_full": (False, lambda c: f"an overhead image of {c}, with distinct structural features, large and prominent"),
    "tdm_tiled": (True, lambda c: f"an overhead image of {c}, with distinct structural features, large and prominent")
}

data, results = collect(), {}
for name, (tiled, prompt) in methods.items():
    print("\n[" + name + "]")
    all_preds, all_gts, per_class = [], [], {}
    for cls, samples in data.items():
        preds, gts = [], []
        for path, gt in samples:
            image = Image.open(path).convert("RGB")
            preds.append(detect(image, prompt(MAP[cls]), tiled, processor, model))
            gts.append(gt)
        per_class[cls] = evaluate(preds, gts)
        all_preds.extend(preds); all_gts.extend(gts)
        print(cls, per_class[cls])
    results[name] = {"per_class": per_class,
                     "overall": evaluate(all_preds, all_gts)}
    print("overall:", results[name]["overall"])

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("\nSaved:", OUT)
