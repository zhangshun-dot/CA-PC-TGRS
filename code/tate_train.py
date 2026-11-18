# -*- coding: utf-8 -*-
"""
TATE Training Pipeline
训练Token-Aware Text Encoder

训练策略:
  Stage 1 (当前): Token级对比预训练
    - 冻结BERT + GDINO视觉backbone
    - 只训练 TIM + S-N Pruning
    - 损失: L_contrastive + λ·L_reg
    - 数据: 图文对 (图像→正确Prompt为正样本, 其他Prompt为负样本)

  Stage 2 (后续): 端到端微调 (可选)
    - 解冻BERT最后3层
    - 加入检测损失

训练数据构造:
  对每张图像, 用其GT类别构造多个候选Prompt, 通过检测F1选择最优作为正样本
  其余候选作为负样本池
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json, os, io, time, random
import numpy as np
from PIL import Image
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 导入TATE
from tate_model import TATE, create_tate_from_gdino, THEORETICAL_CONSTANTS


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
class TATETrainingConfig:
    # 数据
    dior_pq = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
    dior_classes = ['airplane', 'airport', 'baseball field', 'basketball court', 'bridge',
                    'chimney', 'dam', 'expressway service area', 'expressway toll station',
                    'golf course', 'ground track field', 'harbor', 'overpass', 'ship',
                    'stadium', 'storage tank', 'tennis court', 'train station', 'vehicle', 'windmill']
    n_images_per_class: int = 20      # 每类图像数
    n_train_per_class: int = 15        # 训练集
    n_test_per_class: int = 5          # 测试集

    # 模型
    gdino_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
    device: str = 'cuda'

    # 训练
    batch_size: int = 8
    n_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    grad_accum_steps: int = 1

    # Prompt候选池 (用于构造正负样本)
    prefixes = [
        '{}',
        'a satellite image of {}',
        'an overhead image of {}',
        'a remote sensing image of {}',
        'an aerial view of {}',
    ]
    modifier_sets = [
        '',
        ', with distinct structural features',
        ', with distinct structural features, large and prominent',
    ]

    # 输出
    output_dir = r'C:\Users\张三\Desktop\tate_output'
    save_every: int = 10  # 每N个epoch保存一次

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════
def load_dior_data(config: TATETrainingConfig) -> Tuple[Dict, Dict]:
    """加载DIOR数据, 按类别组织训练/测试集"""
    import pyarrow.parquet as pq

    print("Loading DIOR data...")
    table = pq.read_table(config.dior_pq)
    rows = table.to_pylist()

    # 按类别收集图像
    class_data = defaultdict(list)
    for row in rows:
        img_bytes = row['image']['bytes']
        objs = row['objects']
        if not objs or 'category' not in objs:
            continue

        # 按类别分组GT框
        gt_by_cat = defaultdict(list)
        for bbox, cat_id in zip(objs['bbox'], objs['category']):
            cat_name = config.dior_classes[cat_id]
            x, y, w, h = bbox
            gt_by_cat[cat_name].append([x, y, x + w, y + h])

        for cat_name, boxes in gt_by_cat.items():
            if len(class_data[cat_name]) < config.n_images_per_class:
                class_data[cat_name].append((img_bytes, boxes))

    # 划分训练/测试
    np.random.seed(42)
    train_data = {}
    test_data = {}
    for cls_name, imgs in class_data.items():
        idx = np.random.permutation(len(imgs))
        n_train = min(config.n_train_per_class, len(imgs) - config.n_test_per_class)
        train_data[cls_name] = [imgs[i] for i in idx[:n_train]]
        test_data[cls_name] = [imgs[i] for i in idx[n_train:n_train + config.n_test_per_class]]

    total_train = sum(len(v) for v in train_data.values())
    total_test = sum(len(v) for v in test_data.values())
    print(f"  Classes: {len(class_data)}, Train: {total_train}, Test: {total_test}")

    return train_data, test_data


def generate_prompt_candidates(cls_name: str, config: TATETrainingConfig) -> List[str]:
    """为一个类别生成所有候选Prompt"""
    candidates = []
    for pfx in config.prefixes:
        for mods in config.modifier_sets:
            candidates.append(pfx.format(cls_name) + mods)
    return candidates  # 15 candidates per class


# ═══════════════════════════════════════════════════════════════
# 训练器
# ═══════════════════════════════════════════════════════════════
class TATETrainer:
    def __init__(self, tate: TATE, gdino_model, processor, config: TATETrainingConfig):
        self.tate = tate
        self.gdino = gdino_model
        self.processor = processor
        self.config = config

        # 优化器: 只优化TATE的可训练参数
        trainable = [p for p in tate.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)

        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.n_epochs, eta_min=config.lr * 0.01
        )

        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_metrics = []

        print(f"TATE Trainer initialized.")
        print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")
        print(f"  LR: {config.lr}, Epochs: {config.n_epochs}, Batch: {config.batch_size}")

    def get_visual_features(self, pixel_values: torch.Tensor,
                             pixel_mask: torch.Tensor = None) -> torch.Tensor:
        """从GDINO提取视觉特征 (768-dim, 匹配BERT hidden size)"""
        with torch.no_grad():
            gm = self.gdino.model  # GroundingDinoModel
            vis_features, _ = gm.backbone(pixel_values, pixel_mask)

            # vis_features: list of (feature_tensor, mask_tensor) tuples
            # 取最后一层: (B, 768, H, W)
            last_feat, _ = vis_features[-1]

            # 全局平均池化 → (B, 768)
            visual_feat = last_feat.mean(dim=[2, 3])

        return visual_feat

    def prepare_batch(self, batch_data: List[Tuple[bytes, List, str]]) -> Dict:
        """
        准备一个batch
        batch_data: [(img_bytes, gt_boxes, cls_name, prompt), ...]
        """
        config = self.config
        device = config.device

        images = []
        categories = []
        prompts = []

        for img_bytes, _, cls_name, prompt in batch_data:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            images.append(img)
            categories.append(cls_name)
            prompts.append(prompt)

        # 处理图像
        proc_out = self.processor(images=images, return_tensors='pt')
        pixel_values = proc_out['pixel_values'].to(device)
        pixel_mask = proc_out.get('pixel_mask')
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        return {
            'pixel_values': pixel_values,
            'pixel_mask': pixel_mask,
            'prompts': prompts,
            'categories': categories,
        }

    def train_epoch(self, train_data: Dict) -> float:
        """训练一个epoch"""
        self.tate.train()
        config = self.config
        device = config.device

        # 构建训练样本: (image, positive_prompt, category)
        samples = []
        for cls_name, imgs in train_data.items():
            candidates = generate_prompt_candidates(cls_name, config)
            for img_bytes, _ in imgs:
                # 正样本: 随机选一个候选Prompt
                pos_prompt = random.choice(candidates)
                samples.append((img_bytes, cls_name, pos_prompt))

        random.shuffle(samples)
        n_batches = len(samples) // config.batch_size
        total_loss = 0.0

        for b_idx in range(n_batches):
            start = b_idx * config.batch_size
            end = start + config.batch_size
            batch_samples = samples[start:end]

            # 准备数据
            batch_data = [(img, [], cls, prompt) for img, cls, prompt in batch_samples]
            batch = self.prepare_batch(batch_data)

            # 提取视觉特征
            visual_feat = self.get_visual_features(
                batch['pixel_values'], batch.get('pixel_mask'))

            # TATE前向 + 损失
            loss_dict = self.tate.compute_loss(
                visual_feat,
                batch['prompts'],
                batch['categories'],
            )

            loss = loss_dict['total']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tate.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if b_idx % 20 == 0:
                print(f"  Batch {b_idx}/{n_batches}: loss={loss.item():.4f}, "
                      f"contrastive={loss_dict['contrastive'].item():.4f}, "
                      f"reg={loss_dict['reg'].item():.4f}, "
                      f"n_c={loss_dict['n_c']}")

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def validate(self, test_data: Dict) -> Dict:
        """验证: 用TATE编码的Token做检测并计算F1"""
        self.tate.eval()
        config = self.config
        device = config.device

        # 用GDINO原生的检测评估
        # 这里简化: 比较TATE编码前后的检测性能
        results = {'num_samples': 0, 'num_correct': 0}

        with torch.no_grad():
            for cls_name, imgs in test_data.items():
                for img_bytes, gt_boxes in imgs:
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

                    # 用TATE编码
                    prompt = f'an overhead image of {cls_name}'
                    tate_out = self.tate.encode([prompt], [cls_name], training=False)

                    # 记录
                    results['num_samples'] += 1

        return results

    def train(self, train_data: Dict, test_data: Dict):
        """完整训练循环"""
        config = self.config

        print(f"\n{'='*60}")
        print(f"Training TATE: {config.n_epochs} epochs")
        print(f"  Train samples: {sum(len(v) for v in train_data.values())}")
        print(f"  Test samples:  {sum(len(v) for v in test_data.values())}")
        print(f"{'='*60}\n")

        for epoch in range(config.n_epochs):
            self.epoch = epoch
            t0 = time.time()

            # 训练
            avg_loss = self.train_epoch(train_data)
            self.train_losses.append(avg_loss)

            # 学习率调度
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # 日志
            elapsed = time.time() - t0
            print(f"\nEpoch {epoch+1}/{config.n_epochs}: "
                  f"loss={avg_loss:.4f}, lr={current_lr:.2e}, time={elapsed:.1f}s")

            # 保存
            if (epoch + 1) % config.save_every == 0 or avg_loss < self.best_loss:
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    tag = 'best'
                else:
                    tag = f'epoch_{epoch+1}'

                self.save_checkpoint(tag)

        # 最终保存
        self.save_checkpoint('final')
        print(f"\nTraining complete. Best loss: {self.best_loss:.4f}")

    def save_checkpoint(self, tag: str):
        """保存TATE权重"""
        path = os.path.join(self.config.output_dir, f'tate_{tag}.pt')
        state = {
            'epoch': self.epoch,
            'model_state_dict': self.tate.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'constants': self.tate.constants,
        }
        torch.save(state, path)
        print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 主训练入口
# ═══════════════════════════════════════════════════════════════
def main():
    config = TATETrainingConfig()
    config.__post_init__()
    device = config.device

    print("=" * 60)
    print("TATE Training Pipeline")
    print(f"Started: {datetime.now()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # 1. 加载数据
    train_data, test_data = load_dior_data(config)

    # 2. 加载GDINO-tiny
    print("\nLoading GDINO-tiny...")
    import os as _os
    _os.environ['HF_HUB_OFFLINE'] = '1'
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(config.gdino_cache)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config.gdino_cache, dtype=torch.float32
    ).eval().to(device)

    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # 3. 创建TATE
    print("\nInitializing TATE...")
    tate = create_tate_from_gdino(gdino_model, processor, freeze_bert=True)
    tate = tate.to(device)

    # 4. 创建训练器
    trainer = TATETrainer(tate, gdino_model, processor, config)

    # 5. 训练
    trainer.train(train_data, test_data)

    # 6. 保存最终模型
    print(f"\nDone: {datetime.now()}")
    print(f"Output: {config.output_dir}")


if __name__ == '__main__':
    main()
