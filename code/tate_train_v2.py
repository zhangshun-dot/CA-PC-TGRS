# -*- coding: utf-8 -*-
"""
TATE Training v2: Detection-F1-guided Token Importance Learning

核心改进: 不用随机特征的对比损失, 而是用GDINO真实检测F1作为监督信号

训练策略:
  对每个类别, 用15个候选Prompt在训练集上跑GDINO检测
  → 得到每个Prompt的检测F1
  → TATE学习预测: 哪些Token组合会产生高F1?
  → 目标: TIM权重与真实检测F1排序一致

监督信号构造:
  对每个(图像, Prompt)组合:
    - 计算GDINO检测F1 (Teacher)
    - TATE输出Token重要性权重
    - 损失: 权重分布应该最大化教师F1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json, os, io, time, random, sys
import numpy as np
from PIL import Image
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple
import pyarrow.parquet as pq

sys.path.insert(0, '.')
from tate_model import TATE, create_tate_from_gdino, THEORETICAL_CONSTANTS

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
class Config:
    dior_pq = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
    dior_classes = ['airplane','airport','baseball field','basketball court','bridge',
                    'chimney','dam','expressway service area','expressway toll station',
                    'golf course','ground track field','harbor','overpass','ship',
                    'stadium','storage tank','tennis court','train station','vehicle','windmill']
    gdino_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
    output_dir = r'C:\Users\张三\Desktop\tate_output'

    device = 'cuda'
    n_epochs = 30
    batch_size = 4
    lr = 1e-3
    n_teacher_imgs_per_class = 5   # 每个类别用N张图做教师评估
    n_train_imgs_per_class = 15    # 每个类别用M张图训练
    n_candidates_per_class = 5     # 评估的候选Prompt数 (减少计算量)

    prefixes = ['a satellite image of {}', 'an overhead image of {}', 'a remote sensing image of {}']
    modifier_sets = ['', ', with distinct structural features',
                     ', with distinct structural features, large and prominent']

# ═══════════════════════════════════════════════════════════════
# 数据 + 教师评估
# ═══════════════════════════════════════════════════════════════
def iou(b1, b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def compute_f1(preds, gts, thresh=0.25):
    tp=fp=fn=0
    matched=set()
    for pb in preds:
        best=0;bg=-1
        for gi,gb in enumerate(gts):
            if gi in matched: continue
            v=iou(pb,gb)
            if v>best: best=v;bg=gi
        if best>=thresh: matched.add(bg);tp+=1
        else: fp+=1
    fn=len(gts)-len(matched)
    p=tp/(tp+fp) if(tp+fp)else 0;r=tp/(tp+fn) if(tp+fn)else 0
    return 2*p*r/(p+r) if(p+r)else 0

def build_candidates(cls_name, config):
    cands = []
    for pfx in config.prefixes:
        for mods in config.modifier_sets:
            cands.append(pfx.format(cls_name) + mods)
    return cands[:config.n_candidates_per_class]

def load_data_and_teacher_f1(config):
    """加载DIOR数据并计算每个Prompt的教师F1"""
    print("Loading DIOR...")
    table = pq.read_table(config.dior_pq); rows = table.to_pylist()
    class_data = defaultdict(list)
    for row in rows:
        img_bytes = row['image']['bytes']
        objs = row['objects']
        if not objs or 'category' not in objs: continue
        gt_by_cat = defaultdict(list)
        for bbox, cat_id in zip(objs['bbox'], objs['category']):
            x,y,w,h = bbox; gt_by_cat[config.dior_classes[cat_id]].append([x,y,x+w,y+h])
        for cat, boxes in gt_by_cat.items():
            if len(class_data[cat]) < config.n_train_imgs_per_class:
                class_data[cat].append((img_bytes, boxes))

    np.random.seed(42)
    teacher_data = {}
    train_data = {}
    for cls_name, imgs in class_data.items():
        idx = np.random.permutation(len(imgs))
        n_teacher = min(config.n_teacher_imgs_per_class, len(imgs) - 5)
        teacher_data[cls_name] = [imgs[i] for i in idx[:n_teacher]]
        train_data[cls_name] = [imgs[i] for i in idx[n_teacher:]]

    # 加载GDINO计算教师F1
    print("Loading GDINO for teacher F1...")
    import os as _os; _os.environ['HF_HUB_OFFLINE'] = '1'
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    proc = AutoProcessor.from_pretrained(config.gdino_cache)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
        config.gdino_cache, dtype=torch.float32).eval().to(config.device)

    def detect(img_bytes, prompt):
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(config.device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad():
            o = gdino(**inputs)
        target = torch.tensor([img.size[::-1]]).to(config.device)
        r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
        return r['boxes'].cpu().tolist()

    print("Computing teacher F1 scores...")
    teacher_scores = {}
    for cls_name, imgs in teacher_data.items():
        cands = build_candidates(cls_name, config)
        cls_scores = {}
        for cand in cands:
            f1s = []
            for img_bytes, gt_boxes in imgs:
                preds = detect(img_bytes, cand)
                f1s.append(compute_f1(preds, [(b[0],b[1],b[2],b[3]) for b in gt_boxes]))
            cls_scores[cand] = float(np.mean(f1s))
        teacher_scores[cls_name] = cls_scores
        best = max(cls_scores, key=cls_scores.get)
        print(f"  {cls_name}: best={best[:50]}... F1={cls_scores[best]:.4f}, "
              f"worst F1={min(cls_scores.values()):.4f}")

    # 保存教师分数供后续使用
    with open(os.path.join(config.output_dir, 'teacher_scores.json'), 'w') as f:
        json.dump(teacher_scores, f, indent=2, ensure_ascii=False)

    return train_data, teacher_scores, proc, gdino

# ═══════════════════════════════════════════════════════════════
# Training v2: F1-Ranking Loss
# ═══════════════════════════════════════════════════════════════
class TATETrainerV2:
    def __init__(self, tate, teacher_scores, config):
        self.tate = tate
        self.teacher_scores = teacher_scores
        self.config = config

        trainable = [p for p in tate.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    def compute_ranking_loss(self, cls_name, batch_prompts):
        """
        F1排序损失: TATE的Token权重分布应该与教师F1排序一致

        对每个类别, 有K个候选Prompt, 每个有教师F1分数
        TATE输出每个Prompt的Token重要性权重W
        损失 = 排序一致性损失 (ListMLE / RankNet风格)
        """
        if cls_name not in self.teacher_scores:
            return torch.tensor(0.0, device=self.tate.logit_scale.device)

        scores = self.teacher_scores[cls_name]
        available = [p for p in batch_prompts if p in scores]
        if len(available) < 2:
            return torch.tensor(0.0, device=self.tate.logit_scale.device)

        # 获取教师F1分数
        teacher_f1s = torch.tensor([scores[p] for p in available],
                                    device=self.tate.logit_scale.device)

        # TATE编码
        enc = self.tate.encode(available, [cls_name]*len(available), training=True)
        weights = enc['weights'].squeeze(-1)  # (B, k)

        # TATE预测的"Prompt质量": Token权重的加权平均
        # 高质量Prompt → 少数Token有高权重 (集中)
        # 低质量Prompt → 权重分散 (均匀)
        # 用权重的标准差的负数作为质量指标 (越集中=越好)
        tate_score = weights.std(dim=-1)  # (B,) 权重标准差越大 → 越集中 → 越好

        # 排序损失: 用pairwise hinge loss
        # 对每对 (i,j), 如果 teacher_f1[i] > teacher_f1[j],
        # 则 tate_score[i] 应该 > tate_score[j]
        B = len(available)
        loss = 0.0
        n_pairs = 0
        for i in range(B):
            for j in range(B):
                if teacher_f1s[i] > teacher_f1s[j] + 0.01:  # 显著差异
                    # tate_score[i] 应该 > tate_score[j]
                    diff = tate_score[j] - tate_score[i]  # 正=错误排序
                    loss += F.relu(0.1 + diff)  # margin=0.1
                    n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs

        # 额外约束: 整体权重不要太均匀 (鼓励稀疏性)
        # 熵正则化: -H(weights) 即鼓励低熵(高确定性)
        w_norm = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        entropy = -(w_norm * torch.log(w_norm + 1e-8)).sum(dim=-1).mean()
        sparsity_bonus = 0.01 * entropy  # 鼓励低熵 = 更集中的权重

        return loss + sparsity_bonus

    def train_epoch(self, train_data):
        self.tate.train()
        config = self.config

        samples = []
        for cls_name, imgs in train_data.items():
            cands = build_candidates(cls_name, config)
            for img_bytes, _ in imgs:
                for cand in cands:
                    samples.append((cls_name, cand))

        random.shuffle(samples)
        n_batches = len(samples) // config.batch_size
        total_loss = 0.0

        for b_idx in range(n_batches):
            start = b_idx * config.batch_size
            end = start + config.batch_size
            batch = samples[start:end]

            # 按类别分组
            by_class = defaultdict(list)
            for cls_name, prompt in batch:
                by_class[cls_name].append(prompt)

            # 每类独立计算排序损失
            batch_loss = 0.0
            for cls_name, prompts in by_class.items():
                batch_loss += self.compute_ranking_loss(cls_name, prompts)

            self.optimizer.zero_grad()
            if isinstance(batch_loss, torch.Tensor) and batch_loss.requires_grad:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.tate.parameters(), 1.0)
                self.optimizer.step()
                total_loss += batch_loss.item()

            if b_idx % 50 == 0:
                print(f"  Batch {b_idx}/{n_batches}: loss={batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss:.4f}")

        return total_loss / max(n_batches, 1)

    def train(self, train_data, n_epochs):
        for epoch in range(n_epochs):
            t0 = time.time()
            avg_loss = self.train_epoch(train_data)
            print(f"Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, time={time.time()-t0:.1f}s")

            if (epoch + 1) % 10 == 0:
                path = os.path.join(self.config.output_dir, f'tate_v2_epoch{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.tate.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, path)
                print(f"  Saved: {path}")

        final_path = os.path.join(self.config.output_dir, 'tate_v2_final.pt')
        torch.save({'model_state_dict': self.tate.state_dict()}, final_path)
        print(f"Final: {final_path}")

# ═══════════════════════════════════════════════════════════════
# Evaluation: TATE vs Baseline detection F1
# ═══════════════════════════════════════════════════════════════
def evaluate_tate_vs_baseline(tate, gdino_model, processor, config, n_imgs_per_class=5):
    """对比 TATE增强的Prompt vs 标准Prompt 的检测F1"""
    print("\n" + "="*60)
    print("Evaluating TATE vs Baseline")
    print("="*60)

    table = pq.read_table(config.dior_pq); rows = table.to_pylist()
    class_test = defaultdict(list)
    for row in rows:
        img_bytes = row['image']['bytes']
        objs = row['objects']
        if not objs or 'category' not in objs: continue
        gt_by_cat = defaultdict(list)
        for bbox, cat_id in zip(objs['bbox'], objs['category']):
            x,y,w,h = bbox; gt_by_cat[config.dior_classes[cat_id]].append([x,y,x+w,y+h])
        for cat, boxes in gt_by_cat.items():
            if len(class_test[cat]) < n_imgs_per_class:
                class_test[cat].append((img_bytes, boxes))

    device = config.device

    def detect_f1(img_bytes, prompt):
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = processor(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad():
            o = gdino_model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = processor.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
        preds = r['boxes'].cpu().tolist()
        return compute_f1(preds, [(b[0],b[1],b[2],b[3]) for b in gt_boxes])

    # TATE: 用TIM权重选择最优Prompt
    results = {'baseline_overhead': [], 'baseline_satellite': [], 'tate_best': []}

    for cls_name, imgs in class_test.items():
        cands = build_candidates(cls_name, config)

        # TATE: 对每个候选计算TIM权重集中度, 选最集中的
        with torch.no_grad():
            best_cand = None
            best_score = -1
            for cand in cands:
                enc = tate.encode([cand], [cls_name], training=False)
                w = enc['weights'].squeeze(-1)  # (1, k)
                score = w.std().item()  # 权重标准差 = 集中度
                if score > best_score:
                    best_score = score
                    best_cand = cand

        for img_bytes, gt_boxes in imgs:
            gts = [(b[0],b[1],b[2],b[3]) for b in gt_boxes]
            results['baseline_overhead'].append(detect_f1(img_bytes, f'an overhead image of {cls_name}'))
            results['baseline_satellite'].append(detect_f1(img_bytes, f'a satellite image of {cls_name}'))
            results['tate_best'].append(detect_f1(img_bytes, best_cand))

    print(f"\n{'Method':<25} {'Mean F1':>10} {'Median F1':>10}")
    print("-"*45)
    for method, f1s in results.items():
        mean_f1 = np.mean(f1s)
        median_f1 = np.median(f1s)
        print(f"{method:<25} {mean_f1:>10.4f} {median_f1:>10.4f}")

    return results

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print("="*60)
    print("TATE Training v2: Detection-F1-guided")
    print(f"Started: {datetime.now()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)

    # 1. 加载数据 + 教师F1
    train_data, teacher_scores, processor, gdino_model = load_data_and_teacher_f1(config)

    # 2. 创建TATE
    print("\nInitializing TATE...")
    tate = create_tate_from_gdino(gdino_model, processor, freeze_bert=True)
    tate = tate.to(config.device)

    # 3. 训练
    trainer = TATETrainerV2(tate, teacher_scores, config)
    trainer.train(train_data, config.n_epochs)

    # 4. 评估
    eval_results = evaluate_tate_vs_baseline(tate, gdino_model, processor, config)

    # 5. 保存
    with open(os.path.join(config.output_dir, 'eval_results.json'), 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in eval_results.items()}, f, indent=2)
    print(f"\nDone: {datetime.now()}")

if __name__ == '__main__':
    main()
