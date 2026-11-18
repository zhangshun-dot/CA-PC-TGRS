# -*- coding: utf-8 -*-
"""GDINO-Calibrated Verifier: 用历史 GDINO 结果训练一个"见过反馈"的识别器

与纯 CPS 的区别:
  - CPS: 只看 prompt 文本 → 预测 OFAT ΔF1 (不知道 GDINO 实际会怎么判)
  - Verifier: 看 prompt 文本 + 图像特征 + prompt 结构 → 预测 GDINO_DR
             (从 74 个历史 GDINO 结果中学到了 CLIP sim 不等于 GDINO DR)

特征设计:
  1. CLIP text embedding (512-d) — prompt 语义
  2. CLIP image-prompt sim stats (mean/std) — 图像匹配度 (但经过校准!)
  3. Prompt 结构: word count, modifier count, prefix type
  4. CPS score — 已有的强特征
  5. Class embedding — 类间差异

关键创新: 不是直接用 CLIP sim 做判断, 而是学一个从 CLIP sim + 其他特征
到 GDINO DR 的非线性映射。这样 CLIP sim 的"偏见"(喜欢多修饰词) 可以被
其他特征(如 modifier count 的惩罚项) 校正。

闭环:
  CPS 提名 top-8 → Verifier 预测 GDINO_DR → 选 top-3 → GDINO 终审
  GDINO 终审结果 → 加入训练集 → Verifier 自我进化

Usage:
    python gdino_verifier.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, glob, random, pickle
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
DATA_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
DIOR_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'

AXIS_MODIFIERS = {
    'granularity': 'with fine details',
    'spatial': 'in its natural setting',
    'geometric': 'with distinct structural features',
    'scale': 'large and prominent',
    'spectral': 'with natural coloring',
    'texture': 'with clearly visible surface features',
}

MODIFIER_VARIANTS = {
    'granularity': ['with fine details', 'showing intricate details',
                    'with sharp edges', 'featuring detailed structures'],
    'spatial': ['in its natural setting', 'in context of surroundings',
                'within its environment', 'surrounded by typical terrain'],
    'geometric': ['with distinct structural features', 'showing clear geometry',
                  'with well-defined shape', 'featuring characteristic outlines'],
    'scale': ['large and prominent', 'clearly visible in scale',
              'dominating the view', 'at full resolution'],
    'spectral': ['with natural coloring', 'in true color',
                 'with realistic tones', 'showing natural hues'],
    'texture': ['with clearly visible surface features', 'showing surface texture',
                'with visible material patterns', 'featuring surface details'],
}

PREFIXES = [
    'an overhead image of', 'a satellite image of', 'a photo of',
    'a remote sensing image of', 'an aerial view of',
]

AXIS_ORDER = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']


# ══════════════════════════════════════════════════════════════════
# 特征提取器
# ══════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """从 (class, prompt, images) 提取特征向量。"""

    def __init__(self):
        import clip as clip_module
        clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
        self.clip_model, self.preprocess = clip_module.load(clip_path, device=DEVICE)
        self.clip_model.eval()
        self.tokenize = clip_module.tokenize

        # 加载 CPS
        ckpt = torch.load(SAVE_DIR / 'cps_model_notanh.pt', map_location=DEVICE)
        from cps_adapter import CLIPPromptScorer
        self.cps = CLIPPromptScorer(
            d_input=ckpt['config']['d_input'],
            hidden_dims=ckpt['config']['hidden_dims'],
            dropout=ckpt['config']['dropout'],
        )
        self.cps.load_state_dict(ckpt['state_dict'])
        self.cps.eval()
        self.cps.to(DEVICE)

        # Class name → index
        from ofat_knowledge_base import OFATKnowledgeBase
        kb = OFATKnowledgeBase()
        all_classes = sorted(kb.dior_classes)
        self.class_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.n_classes = len(all_classes)

    def get_clip_text_embedding(self, prompt):
        """CLIP text embedding (512-d, 归一化)。"""
        tok = self.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tok)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().float().numpy()

    def get_clip_image_sims(self, images, prompt):
        """计算每张图的 CLIP cos_sim(image, prompt)。

        Returns:
            np.ndarray: [mean_sim, std_sim, min_sim, max_sim]
        """
        if not images:
            return np.array([0.0, 0.0, 0.0, 0.0])

        text_tok = self.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            text_feat = self.clip_model.encode_text(text_tok)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        sims = []
        for img in images[:10]:  # 最多 10 张
            img_input = self.preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_feat = self.clip_model.encode_image(img_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sim = (img_feat @ text_feat.T).item()
            sims.append(sim)

        sims = np.array(sims)
        return np.array([sims.mean(), sims.std(), sims.min(), sims.max()])

    def get_prompt_structure(self, prompt):
        """Prompt 结构特征。

        Returns:
            np.ndarray: [word_count, modifier_count, char_count,
                         is_overhead, is_satellite, is_photo, is_remote, is_aerial]
        """
        words = prompt.split()
        word_count = len(words)

        # 统计修饰词
        modifier_count = 0
        lower = prompt.lower()
        for axis, variants in MODIFIER_VARIANTS.items():
            for v in variants:
                if v.lower() in lower:
                    modifier_count += 1
                    break

        char_count = len(prompt)

        # Prefix 类型 (one-hot)
        is_overhead = 1 if 'overhead' in lower else 0
        is_satellite = 1 if 'satellite' in lower else 0
        is_photo = 1 if lower.startswith('a photo of') else 0
        is_remote = 1 if 'remote sensing' in lower else 0
        is_aerial = 1 if 'aerial' in lower else 0

        return np.array([word_count, modifier_count, char_count,
                        is_overhead, is_satellite, is_photo, is_remote, is_aerial],
                       dtype=np.float32)

    def get_cps_score(self, prompt):
        """CPS 预测的 ΔF1。"""
        tok = self.tokenize([prompt], truncate=True).to(DEVICE)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tok)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            score = self.cps(emb.float().to(DEVICE)).item()
        return score

    def get_class_features(self, class_name):
        """类特征: one-hot + CLIP 嵌入。"""
        # CLIP 类名嵌入
        prompt = f'a photo of {class_name}'
        cls_emb = self.get_clip_text_embedding(prompt)
        return cls_emb  # (512,)

    def extract(self, class_name, prompt, images=None):
        """提取完整特征向量。

        Args:
            class_name: 类名
            prompt: prompt 文本
            images: List[PIL.Image] 或 None

        Returns:
            np.ndarray: 特征向量 (n_features,)
        """
        features = []

        # 1. CLIP text embedding (512-d)
        text_emb = self.get_clip_text_embedding(prompt)
        features.append(text_emb)

        # 2. CLIP image-prompt sim stats (4-d)
        if images:
            sim_stats = self.get_clip_image_sims(images, prompt)
        else:
            sim_stats = np.array([0.25, 0.02, 0.20, 0.30])  # 默认值
        features.append(sim_stats)

        # 3. Prompt structure (8-d)
        struct = self.get_prompt_structure(prompt)
        features.append(struct)

        # 4. CPS score (1-d)
        cps_score = np.array([self.get_cps_score(prompt)])
        features.append(cps_score)

        # 5. Class embedding (512-d)
        cls_emb = self.get_class_features(class_name)
        features.append(cls_emb)

        return np.concatenate(features)  # 512 + 4 + 8 + 1 + 512 = 1037-d

    def extract_simple(self, class_name, prompt, images=None):
        """提取简化特征 (可解释 + 轻量)。

        Returns:
            np.ndarray: 简化特征向量
        """
        features = []

        # CPS score
        features.append(self.get_cps_score(prompt))

        # Prompt structure
        struct = self.get_prompt_structure(prompt)
        features.extend(struct)

        # CLIP sim stats
        if images:
            sim_stats = self.get_clip_image_sims(images, prompt)
        else:
            sim_stats = np.array([0.25, 0.02, 0.20, 0.30])
        features.extend(sim_stats)

        # Class index (one-hot, 20-d)
        cls_idx = self.class_to_idx.get(class_name, 0)
        cls_onehot = np.zeros(self.n_classes)
        cls_onehot[cls_idx] = 1.0
        features.extend(cls_onehot)

        return np.array(features, dtype=np.float32)  # 1 + 8 + 4 + 20 = 33-d


# ══════════════════════════════════════════════════════════════════
# GDINO-Calibrated Verifier
# ══════════════════════════════════════════════════════════════════

class GDINOVerifier:
    """用历史 GDINO 结果训练的识别器。

    支持两种模式:
      - 'gbdt': GradientBoostingRegressor (稳定, 小数据友好)
      - 'ridge': Ridge regression (简单, 可解释)
    """

    def __init__(self, extractor, mode='gbdt'):
        self.extractor = extractor
        self.mode = mode
        self.scaler = StandardScaler()

        if mode == 'gbdt':
            self.model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                min_samples_leaf=3, random_state=42
            )
        elif mode == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif mode == 'classifier':
            # 二分类: DR > baseline 视为 "好"
            self.model = LogisticRegression(C=1.0, class_weight='balanced')
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.is_trained = False

    def fit(self, X, y):
        """训练识别器。

        Args:
            X: (n_samples, n_features) 特征矩阵
            y: (n_samples,) GDINO detection rates
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # 评估
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            corr = np.corrcoef(y, y_pred)[0, 1]
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            print(f"  训练完成: R²={r2:.4f}, r={corr:.4f}, RMSE={rmse:.4f}")

        # 特征重要性 (仅 GBDT)
        if self.mode == 'gbdt' and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_idx = np.argsort(importances)[-10:][::-1]
            print(f"  Top-5 特征重要性:")
            for i in top_idx[:5]:
                print(f"    feat[{i}] = {importances[i]:.4f}")

    def predict(self, X):
        """预测 GDINO detection rate。

        Args:
            X: (n_samples, n_features)

        Returns:
            np.ndarray: 预测 DR
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_single(self, class_name, prompt, images=None):
        """单次预测。"""
        feats = self.extractor.extract_simple(class_name, prompt, images)
        return self.predict(feats.reshape(1, -1))[0]


# ══════════════════════════════════════════════════════════════════
# 训练数据准备
# ══════════════════════════════════════════════════════════════════

def prepare_training_data(extractor, n_images_per_class=10):
    """从历史 GDINO 结果 + 实时 CLIP 特征准备训练数据。

    对每个 (class, prompt, GDINO_DR) 数据点:
      1. 加载该类图像
      2. 提取 CLIP image-prompt sim stats
      3. 构造完整特征向量

    Returns:
        X: (n_samples, n_features)
        y: (n_samples,)
        metadata: List[Dict]
    """
    # 加载历史数据
    data_path = DATA_DIR / 'gdino_training_data.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"加载 {len(raw_data)} 个历史数据点")

    # 预加载图像 (按类缓存)
    image_cache = {}
    for d in raw_data:
        cls_name = d['class']
        if cls_name not in image_cache:
            paths = sorted(glob.glob(
                os.path.join(DIOR_DIR, f"{cls_name.replace(' ','_')}_*.jpg")
            ))[:n_images_per_class]
            if not paths:
                paths = sorted(glob.glob(
                    os.path.join(DIOR_DIR, f"{cls_name}_*.jpg")
                ))[:n_images_per_class]
            image_cache[cls_name] = [Image.open(p).convert('RGB') for p in paths]

    X_list, y_list, meta_list = [], [], []

    for i, d in enumerate(raw_data):
        cls_name = d['class']
        prompt = d['prompt']
        dr = d['dr']
        images = image_cache.get(cls_name, [])

        try:
            feats = extractor.extract_simple(cls_name, prompt, images)
            X_list.append(feats)
            y_list.append(dr)
            meta_list.append(d)

            if i % 20 == 0:
                print(f"  [{i}/{len(raw_data)}] {cls_name}: DR={dr:.3f}")
        except Exception as e:
            print(f"  跳过 [{i}]: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"\n训练数据: X={X.shape}, y={y.shape}")
    print(f"  DR 分布: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")

    return X, y, meta_list


# ══════════════════════════════════════════════════════════════════
# GDINO 评估
# ══════════════════════════════════════════════════════════════════

def gdino_detect_rate(class_name, prompt, n_images=20, threshold=0.25):
    """GDINO 检测命中率。"""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
    gd_proc = AutoProcessor.from_pretrained(gd_cache)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gd_cache, dtype=torch.float32
    ).eval().to(DEVICE)

    paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{class_name.replace(' ','_')}_*.jpg")))[:n_images]
    if not paths:
        paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{class_name}_*.jpg")))[:n_images]
    images = [Image.open(p).convert('RGB') for p in paths]
    if not images:
        return 0.0

    detections = []
    for img in images:
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(
            o, threshold=threshold, target_sizes=ts
        )[0]
        detections.append(len(r['boxes']) > 0)
    return float(np.mean(detections))


# ══════════════════════════════════════════════════════════════════
# 候选生成
# ══════════════════════════════════════════════════════════════════

def generate_candidates(class_name, kb, n_total=40):
    """生成候选 prompt。"""
    candidates = set()

    for prefix in PREFIXES:
        candidates.add(f'{prefix} {class_name}')

    for axis in AXIS_ORDER:
        for mod in MODIFIER_VARIANTS[axis]:
            candidates.add(f'an overhead image of {class_name}, {mod}')

    recs = kb.get_recommended_axes('remote_sensing', class_name)
    top_axes = [a for a, _ in recs[:3]]
    if len(top_axes) >= 2:
        for prefix in PREFIXES[:3]:
            for i, a1 in enumerate(top_axes):
                for a2 in top_axes[i+1:]:
                    m1, m2 = AXIS_MODIFIERS[a1], AXIS_MODIFIERS[a2]
                    candidates.add(f'{prefix} {class_name}, {m1}, {m2}')

    all_mods = []
    for axis in AXIS_ORDER:
        all_mods.extend(MODIFIER_VARIANTS[axis])

    rng = random.Random(hash(class_name) % 10000)
    while len(candidates) < n_total:
        n = rng.randint(1, 3)
        mods = rng.sample(all_mods, min(n, len(all_mods)))
        used_axes = set()
        filtered = []
        for m in mods:
            for axis, variants in MODIFIER_VARIANTS.items():
                if m in variants and axis not in used_axes:
                    filtered.append(m)
                    used_axes.add(axis)
                    break
        prefix = rng.choice(PREFIXES)
        prompt = f'{prefix} {class_name}'
        if filtered:
            prompt += ', ' + ', '.join(filtered)
        candidates.add(prompt)

    return list(candidates)


# ══════════════════════════════════════════════════════════════════
# 主循环: Verifier 驱动搜索
# ══════════════════════════════════════════════════════════════════

def verifier_guided_search(class_name, verifier, extractor, kb, images,
                           baseline_prompt, n_candidates=40):
    """Verifier 驱动的 prompt 搜索。

    流程:
      1. 生成候选
      2. 对每个候选, Verifier 预测 GDINO_DR (不需要跑 GDINO!)
      3. 按预测 DR 排序 → top-3
      4. GDINO 终审 top-3 + baseline + TDM
      5. GDINO 结果反馈到训练集

    Args:
        class_name: 类名
        verifier: GDINOVerifier
        extractor: FeatureExtractor
        kb: OFATKnowledgeBase
        images: List[PIL.Image]
        baseline_prompt: str
        n_candidates: 候选数

    Returns:
        Dict: 搜索结果
    """
    # Step 1: 生成候选
    candidates = generate_candidates(class_name, kb, n_total=n_candidates)
    print(f"  生成 {len(candidates)} 候选")

    # Step 2: Verifier 预测每个候选的 GDINO_DR
    predictions = []
    for prompt in candidates:
        feats = extractor.extract_simple(class_name, prompt, images)
        pred_dr = verifier.predict(feats.reshape(1, -1))[0]
        predictions.append((prompt, float(pred_dr)))

    # 排序
    predictions.sort(key=lambda x: -x[1])
    top3 = predictions[:3]

    print(f"  Verifier top-3:")
    for p, dr in top3:
        print(f"    [pred_DR={dr:.3f}] '{p[:70]}...'")

    # Step 3: GDINO 终审
    tdm_prompt = f'an overhead image of {class_name}'
    recs = kb.get_recommended_axes('remote_sensing', class_name)
    mods = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
    if mods:
        tdm_prompt += ', ' + ', '.join(mods)

    finalists = [baseline_prompt, tdm_prompt]
    for p, _ in top3:
        if p not in finalists:
            finalists.append(p)
    finalists = list(dict.fromkeys(finalists))[:5]

    print(f"  GDINO 终审 {len(finalists)} 候选...")
    gdino_results = {}
    for p in finalists:
        dr = gdino_detect_rate(class_name, p, n_images=len(images))
        gdino_results[p] = dr

    gdino_ranking = sorted(gdino_results.items(), key=lambda x: -x[1])
    best_prompt, best_dr = gdino_ranking[0]
    baseline_dr = gdino_results[baseline_prompt]
    tdm_dr_val = gdino_results[tdm_prompt]
    verifier_best_dr = gdino_results.get(top3[0][0], 0)

    return {
        'class': class_name,
        'baseline_dr': baseline_dr,
        'tdm_dr': tdm_dr_val,
        'verifier_pred_best': top3[0][1],
        'verifier_best_dr': verifier_best_dr,
        'best_dr': best_dr,
        'best_prompt': best_prompt,
        'gdino_ranking': gdino_ranking,
        'verifier_predictions': [(p, float(s)) for p, s in top3],
    }


# ══════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("GDINO-Calibrated Verifier: 从 GDINO 反馈中学习的识别器")
    print("=" * 65)
    print(f"设备: {DEVICE}")

    # ── 初始化 ──
    print("\n初始化特征提取器...")
    extractor = FeatureExtractor()

    # ── 准备训练数据 ──
    print("\n准备训练数据 (历史 GDINO 结果 + CLIP 特征)...")
    X_train, y_train, meta = prepare_training_data(extractor, n_images_per_class=10)

    # ── 训练 Verifier ──
    print(f"\n训练 GDINO Verifier (GBDT, {len(y_train)} 样本)...")
    verifier = GDINOVerifier(extractor, mode='gbdt')
    verifier.fit(X_train, y_train)

    # 交叉验证
    from sklearn.model_selection import cross_val_score
    X_scaled = verifier.scaler.fit_transform(X_train)
    cv_scores = cross_val_score(verifier.model, X_scaled, y_train, cv=5,
                                scoring='neg_mean_squared_error')
    print(f"  5-fold CV RMSE: {np.sqrt(-cv_scores.mean()):.4f} (±{np.sqrt(cv_scores.std()):.4f})")

    # ── 在 DIOR 上测试 ──
    print("\n" + "=" * 60)
    print("Verifier 驱动搜索: DIOR 8 类测试")
    print("=" * 60)

    from ofat_knowledge_base import OFATKnowledgeBase
    kb = OFATKnowledgeBase()

    eval_classes = ['airplane', 'ship', 'stadium', 'bridge', 'harbor',
                    'overpass', 'tennis court', 'windmill']
    n_images = 20

    all_results = {}
    all_new_data = []  # 收集新的 GDINO 结果用于反馈

    for cls in eval_classes:
        print(f"\n{'─'*60}")
        print(f"  {cls}")
        print(f"{'─'*60}")

        baseline_prompt = f'an overhead image of {cls}'

        # 加载图像
        paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{cls.replace(' ','_')}_*.jpg")))[:n_images]
        if not paths:
            paths = sorted(glob.glob(os.path.join(DIOR_DIR, f"{cls}_*.jpg")))[:n_images]
        images = [Image.open(p).convert('RGB') for p in paths]
        if not images:
            continue

        result = verifier_guided_search(
            cls, verifier, extractor, kb, images, baseline_prompt,
            n_candidates=40
        )
        all_results[cls] = result

        # 收集新数据
        for prompt, dr in result['gdino_ranking']:
            all_new_data.append({'class': cls, 'prompt': prompt, 'dr': dr})

        print(f"\n  ▶ {cls}:")
        print(f"    Baseline:        {result['baseline_dr']:.3f}")
        print(f"    Static TDM:      {result['tdm_dr']:.3f}")
        print(f"    Verifier pred:   {result['verifier_pred_best']:.3f} → GDINO: {result['verifier_best_dr']:.3f}")
        print(f"    GDINO best:      {result['best_dr']:.3f}  '{result['best_prompt'][:70]}...'")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")

    # 加载 CPS only 结果对比
    try:
        with open(SAVE_DIR / 'cps_search_results.json', 'r', encoding='utf-8') as f:
            prev = json.load(f)
        prev_cps = {cls: r['cps_best_dr'] for cls, r in prev['final_results'].items()}
        prev_best = {cls: r['gdino_best_dr'] for cls, r in prev['final_results'].items()}
    except:
        prev_cps = {}
        prev_best = {}

    print(f"{'Class':<20s} {'Base':>7s} {'TDM':>7s} {'CPS only':>8s} {'Verifier':>9s} {'Best':>7s}")
    print(f"{'-'*60}")

    total_base, total_tdm, total_cps, total_verifier, total_best = 0, 0, 0, 0, 0
    for cls in eval_classes:
        r = all_results[cls]
        total_base += r['baseline_dr']
        total_tdm += r['tdm_dr']
        total_verifier += r['verifier_best_dr']
        total_best += r['best_dr']

        cps_dr = prev_cps.get(cls, r['verifier_best_dr'])
        total_cps += cps_dr

        print(f"{cls:<20s} {r['baseline_dr']:7.3f} {r['tdm_dr']:7.3f} "
              f"{cps_dr:8.3f} {r['verifier_best_dr']:9.3f} {r['best_dr']:7.3f}")

    n = len(eval_classes)
    print(f"{'-'*60}")
    print(f"{'MEAN':<20s} {total_base/n:7.4f} {total_tdm/n:7.4f} "
          f"{total_cps/n:8.4f} {total_verifier/n:9.4f} {total_best/n:7.4f}")

    # ── 保存 ──
    # 扩增训练数据
    all_training = meta + all_new_data
    with open(SAVE_DIR / 'gdino_training_data_v2.json', 'w', encoding='utf-8') as f:
        json.dump(all_training, f, indent=2, ensure_ascii=False)

    output = {
        'results': {cls: {k: v for k, v in r.items()
                     if k not in ('gdino_ranking', 'verifier_predictions')}
                    for cls, r in all_results.items()},
        'summary': {
            'mean_baseline': total_base / n,
            'mean_tdm': total_tdm / n,
            'mean_cps_only': total_cps / n,
            'mean_verifier': total_verifier / n,
            'mean_best': total_best / n,
        },
        'training_data_size': len(all_training),
    }
    with open(SAVE_DIR / 'gdino_verifier_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {SAVE_DIR / 'gdino_verifier_results.json'}")
    print(f"扩增训练数据: {len(all_training)} 样本 → {SAVE_DIR / 'gdino_training_data_v2.json'}")

    print("\n✅ GDINO-Calibrated Verifier 完成")
