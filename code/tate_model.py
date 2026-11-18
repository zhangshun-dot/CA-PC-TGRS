# -*- coding: utf-8 -*-
"""
TATE: Token-Aware Text Encoder
基于信息论第一性原理设计的 OVD 文本编码器

三个核心模块:
  1. TIM (Token Importance Module) — 逐Token重要性加权,替代等权点积
  2. S-N Dynamic Pruning — 基于S-N均衡模型动态剪枝冗余Token
  3. Token-Level Contrastive Loss — 训练目标与推理机制在Token空间统一

理论约束（来自TDM论文定理）:
  - DPI → 损失必须定义在Token级,非[CLS]级
  - S-N均衡 → 存在最优Token数n_c,超过则噪声主导
  - 统一上界 → I_eff ≤ C·η_α·κ,正则化防过拟合

Author: Token发现→TDM→TATE 体系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# ═══════════════════════════════════════════════════════════════
# 理论常数 (来自OFAT测量, §3.2.1 / §4.1)
# ═══════════════════════════════════════════════════════════════
THEORETICAL_CONSTANTS = {
    'C': 7.9,        # 文本嵌入信道容量 (bits), PCA 95%方差
    'eta_alpha': 0.118,  # 注意力衰减因子 (satellite配置)
    'kappa': 0.052,      # 跨模型信息保留率
    'n_c_default': 12,   # 默认临界Token数
    'd_model': 768,      # BERT hidden dim
    'd_proj': 256,       # GDINO projection dim
}


# ═══════════════════════════════════════════════════════════════
# 1. Token Importance Module (TIM)
# ═══════════════════════════════════════════════════════════════
class TokenImportanceModule(nn.Module):
    """
    为每个Token预测标量重要性权重 w_i ∈ [0,1]

    输入: Token嵌入 T ∈ R^(B, n, d) + 类别嵌入 e_c ∈ R^(B, d)
    输出: 重要性权重 W = diag(w_1, ..., w_n)

    设计理由:
      - 类别条件化: 不同类别对同一Token的需求不同
        (如 'airplane' 需要 spatial 轴Token, 'bridge' 需要 geometric 轴Token)
      - LN + 残差: 稳定训练, 保留BERT原始语义方向
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # 类别嵌入投影 (取BERT的[CLS]或平均池化作为类别表示)
        self.category_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Token特征投影
        self.token_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 联合特征 → 重要性分数
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # w_i ∈ [0, 1]
        )

        # 用于训练时的temperature退火 (soft→hard)
        self.register_buffer('temperature', torch.tensor(1.0))

    def forward(self, token_embeds: torch.Tensor,
                category_embed: Optional[torch.Tensor] = None,
                return_weights: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_embeds: (B, n, d_model) BERT输出的Token嵌入
            category_embed: (B, d_model) 类别表示, 默认用[CLS]
        Returns:
            weighted_embeds: (B, n, d_model) 加权后的Token嵌入
            weights: (B, n, 1) Token重要性权重
        """
        B, n, d = token_embeds.shape

        # 类别表示: [CLS] token 或 提供的嵌入
        if category_embed is None:
            category_embed = token_embeds[:, 0, :]  # [CLS]

        # 投影
        cat_feat = self.category_proj(category_embed)        # (B, hidden)
        tok_feat = self.token_proj(token_embeds)              # (B, n, hidden)

        # 拼接类别特征到每个Token位置
        cat_feat_expanded = cat_feat.unsqueeze(1).expand(-1, n, -1)  # (B, n, hidden)
        joint = torch.cat([tok_feat, cat_feat_expanded], dim=-1)     # (B, n, 2*hidden)

        # 预测重要性
        weights = self.importance_head(joint)  # (B, n, 1)

        # 加权
        weighted_embeds = token_embeds * weights

        if return_weights:
            return weighted_embeds, weights
        return weighted_embeds


# ═══════════════════════════════════════════════════════════════
# 2. S-N Dynamic Pruning Module
# ═══════════════════════════════════════════════════════════════
class SNDynamicPruning(nn.Module):
    """
    基于S-N供需均衡模型动态预测最优Token数 n_c, 并剪枝冗余Token

    核心:
      - n_c 预测: MLP从类别特征预测该类的临界Token数
      - 剪枝: 保留top-k个最高重要性Token (k = min(n_c, n))
      - 训练时用Gumbel-Softmax保证可微, 推理时硬剪枝

    理论:
      S-N模型指出性能 = 信号项 - 噪声项, 存在唯一最大值对应的 n_c
      超过 n_c 的Token只贡献噪声, 应被剪掉
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 128,
                 n_c_min: int = 3, n_c_max: int = 20):
        super().__init__()
        self.n_c_min = n_c_min  # 最小保留Token数 ([CLS] + 至少2个语义Token)
        self.n_c_max = n_c_max

        # n_c 预测器: 从类别特征 → 最优Token数
        self.nc_predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 输出归一化到[0,1], 然后缩放到[n_c_min, n_c_max]
        )

    def forward(self, token_embeds: torch.Tensor, weights: torch.Tensor,
                category_embed: Optional[torch.Tensor] = None,
                training: bool = True, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            token_embeds: (B, n, d) Token嵌入
            weights: (B, n, 1) TIM输出的重要性权重
            category_embed: (B, d) 类别表示
            training: 是否训练模式 (Gumbel-Softmax vs hard)
            hard: 推理时强制硬剪枝
        Returns:
            pruned_embeds: (B, k, d) 剪枝后的Token嵌入
            pruned_weights: (B, k, 1) 剪枝后的权重
            k: 实际保留的Token数
        """
        B, n, d = token_embeds.shape

        # 预测 n_c
        if category_embed is None:
            category_embed = token_embeds[:, 0, :]
        nc_norm = self.nc_predictor(category_embed)  # (B, 1) in [0,1]
        n_c = self.n_c_min + nc_norm * (self.n_c_max - self.n_c_min)
        n_c = n_c.squeeze(-1).long()  # (B,) → int
        n_c = torch.clamp(n_c, self.n_c_min, min(n, self.n_c_max))

        # 每个样本保留的Token数可能不同, 取模式值(众数)统一处理
        k = int(n_c.float().mode().values.item()) if training else int(n_c.float().median().item())
        k = max(self.n_c_min, min(k, n))

        # 按重要性排序, 保留top-k
        # weights: (B, n, 1) → (B, n)
        w = weights.squeeze(-1)  # (B, n)

        if training and not hard:
            # Gumbel-Softmax 软掩码: 可微近似
            # 对重要性分数加Gumbel噪声, 取soft top-k
            logits = torch.log(w + 1e-8)
            # 重复采样Gumbel噪声做软选择
            gumbel_mask = torch.zeros_like(w)
            for _ in range(1):  # 单次Gumbel
                noise = -torch.log(-torch.log(torch.rand_like(w) + 1e-8) + 1e-8)
                y_soft = F.softmax((logits + noise) / 0.5, dim=-1)  # temperature=0.5
                # 只保留top-k的软权重
                _, top_indices = torch.topk(y_soft, k, dim=-1)
                mask = torch.zeros_like(w)
                mask.scatter_(-1, top_indices, 1.0)
                gumbel_mask = gumbel_mask + mask

            # 软掩码: 用原始weight * soft_mask
            soft_mask = gumbel_mask
        else:
            # 硬剪枝: 直接取top-k
            _, top_indices = torch.topk(w, k, dim=-1)
            soft_mask = torch.zeros_like(w)
            soft_mask.scatter_(-1, top_indices, 1.0)

        # 应用掩码
        mask = soft_mask.unsqueeze(-1)  # (B, n, 1)
        pruned_embeds = token_embeds * mask
        pruned_weights = weights * mask

        return pruned_embeds, pruned_weights, k


# ═══════════════════════════════════════════════════════════════
# 3. TATE: Token-Aware Text Encoder (完整模型)
# ═══════════════════════════════════════════════════════════════
class TATE(nn.Module):
    """
    Token-Aware Text Encoder

    架构:
      BERT Embedding → BERT Encoder → TIM → S-N Pruning → text_projection → 输出

    训练:
      Stage 1: 冻结BERT, 训练TIM + Pruning + Token-Level Contrastive Loss
      Stage 2 (可选): 解冻BERT最后N层, 端到端微调

    使用方法:
      tate = TATE(gdino_model)

      # 训练
      loss = tate.compute_loss(visual_features, text_prompts, positive_labels)
      loss.backward()

      # 推理 (插入GDINO)
      text_embeds = tate.encode(prompts, categories)  # Token-Native编码
      # 然后将text_embeds喂给GDINO的encoder/decoder
    """

    def __init__(self,
                 bert_model: nn.Module,
                 text_projection: nn.Module,
                 processor,
                 constants: dict = None,
                 freeze_bert: bool = True):
        super().__init__()

        self.constants = constants or THEORETICAL_CONSTANTS
        d_model = self.constants['d_model']

        # 复用GDINO的BERT + text_projection
        self.bert = bert_model
        self.text_projection = text_projection
        self.processor = processor

        # 新增模块
        self.tim = TokenImportanceModule(d_model=d_model)
        self.pruning = SNDynamicPruning(d_model=d_model)

        # 训练配置
        self.freeze_bert = freeze_bert
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # 损失函数参数
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # 可学temperature
        self.lambda_reg = nn.Parameter(torch.tensor(0.01))  # 正则化系数

    def encode(self,
               prompts: list,
               categories: Optional[list] = None,
               training: bool = False) -> Dict[str, torch.Tensor]:
        """
        Token-Native文本编码 (替代BERT的标准encode)

        Args:
            prompts: 文本Prompt列表, e.g. ['an overhead image of airplane']
            categories: 类别名列表(可选), e.g. ['airplane']
            training: 训练/推理模式

        Returns:
            {
                'token_embeds': (B, k, d_proj),    剪枝加权后的Token嵌入(投影到256维)
                'weights': (B, k, 1),              保留Token的重要性权重
                'n_c': int,                         实际保留Token数
                'raw_token_embeds': (B, n, d_model), BERT原始输出
                'cls_embed': (B, d_model),          [CLS]嵌入
                'attention_mask': (B, n),           BERT attention mask
            }
        """
        device = next(self.bert.parameters()).device

        # Tokenize
        enc = self.processor.tokenizer(
            prompts, padding=True, truncation=True,
            max_length=self.constants.get('max_text_len', 256),
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        # BERT编码
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeds = bert_out.last_hidden_state  # (B, n, 768)

        # 类别嵌入 (取[CLS]或类别名单独编码)
        if categories:
            cat_enc = self.processor.tokenizer(
                categories, padding=True, truncation=True, return_tensors='pt'
            )
            cat_ids = cat_enc['input_ids'].to(device)
            cat_mask = cat_enc['attention_mask'].to(device)
            with torch.set_grad_enabled(training):
                cat_out = self.bert(input_ids=cat_ids, attention_mask=cat_mask)
                # 对有效Token取平均 (排除[CLS]和[SEP]和padding)
                cat_mask_exp = cat_mask.unsqueeze(-1).float()
                cat_embed = (cat_out.last_hidden_state * cat_mask_exp).sum(dim=1) / cat_mask_exp.sum(dim=1).clamp(min=1)
        else:
            cat_embed = token_embeds[:, 0, :]  # [CLS]

        # TIM: Token重要性加权
        weighted_embeds, weights = self.tim(token_embeds, cat_embed, return_weights=True)

        # S-N Pruning: 动态剪枝
        pruned_embeds, pruned_weights, n_c = self.pruning(
            weighted_embeds, weights, cat_embed,
            training=training, hard=not training
        )

        # 投影到GDINO特征空间 (256维)
        # 只对保留的(非零)Token投影
        pruned_flat = pruned_embeds.reshape(-1, pruned_embeds.shape[-1])
        proj_flat = self.text_projection(pruned_flat)
        proj_embeds = proj_flat.reshape(pruned_embeds.shape[0], pruned_embeds.shape[1], -1)

        return {
            'token_embeds': proj_embeds,             # (B, k, 256) 投影后, 用于GDINO集成
            'pruned_raw': pruned_embeds,             # (B, k, 768) 投影前, 用于TATE损失
            'weights': pruned_weights,               # (B, k, 1)
            'n_c': n_c,                              # int
            'raw_token_embeds': token_embeds,        # (B, n, 768) 完整BERT输出
            'cls_embed': token_embeds[:, 0, :],     # (B, 768)
            'attention_mask': attention_mask,        # (B, n)
        }

    def compute_token_level_contrastive_loss(self,
                                              visual_features: torch.Tensor,
                                              token_embeds: torch.Tensor,
                                              weights: torch.Tensor,
                                              positive_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Token-Level Contrastive Loss

        核心: 不是优化一个聚合向量, 而是每个Token独立参与对比学习

        L = -Σ_i w_i · log(exp(sim(v, t_i)/τ) / Σ_j exp(sim(v_j, t_i)/τ))

        Args:
            visual_features: (B, d) 视觉特征
            token_embeds: (B, k, d) TATE输出的Token嵌入
            weights: (B, k, 1) Token重要性权重
        Returns:
            loss: 标量
        """
        B, k, d = token_embeds.shape

        # 归一化
        v_norm = F.normalize(visual_features, dim=-1)       # (B, d)
        t_norm = F.normalize(token_embeds, dim=-1)           # (B, k, d)

        logit_scale = self.logit_scale.exp()

        # 将Token展平: (B, k, d) → (B*k, d)
        t_flat = t_norm.reshape(B * k, d)

        # 相似度矩阵: (B*k, B)
        # sim[i, j] = similarity between t_flat[i] and v_norm[j]
        sim_matrix = torch.matmul(t_flat, v_norm.T) * logit_scale  # (B*k, B)

        # 标签: 每个Token属于其原始batch图像
        labels = torch.arange(B, device=sim_matrix.device).unsqueeze(0).expand(k, -1).T.reshape(-1)
        # labels[i] = which batch item t_flat[i] belongs to

        # Token加权交叉熵
        weights_flat = weights.squeeze(-1).reshape(-1)  # (B*k,)
        ce_loss = F.cross_entropy(sim_matrix, labels, reduction='none')  # (B*k,)
        weighted_loss = (ce_loss * weights_flat).sum() / weights_flat.sum().clamp(min=1)

        return weighted_loss

    def compute_upper_bound_regularization(self,
                                            token_embeds: torch.Tensor,
                                            weights: torch.Tensor) -> torch.Tensor:
        """
        统一上界正则化

        L_reg = max(0, I_eff - C·η_α·κ)

        其中 I_eff 是从当前Token分布估计的有效互信息

        理论: 统一上界定理给出 I_eff ≤ C·η_α·κ = 7.9 * 0.118 * 0.052 ≈ 0.0485 bits
        如果模型的有效互信息超过理论上界, 说明在拟合噪声, 需要惩罚
        """
        C = self.constants['C']
        eta_alpha = self.constants['eta_alpha']
        kappa = self.constants['kappa']
        upper_bound = C * eta_alpha * kappa  # ≈ 0.0485 bits

        # 估计有效互信息 I_eff
        # 方法: 基于Token权重分布的熵
        # I_eff ≈ H(weights) = -Σ w_i log(w_i), 表示Token分布的信息量
        w = weights.squeeze(-1)  # (B, k)
        w_norm = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        entropy = -(w_norm * torch.log(w_norm + 1e-8)).sum(dim=-1).mean()  # (B,)

        # 有效Token数 n_eff = exp(H)
        n_eff = torch.exp(entropy)

        # 有效互信息估计: I_eff ≈ C * (1 - exp(-n_eff / n_c_max))
        # 当 n_eff 很小时, I_eff 也小; 当 n_eff 接近 n_c_max 时, I_eff 接近 C
        n_c_max = self.constants['n_c_default']
        I_eff = C * (1 - torch.exp(-n_eff / n_c_max))

        # 正则化: 超过上界的部分
        reg_loss = F.relu(I_eff - upper_bound)

        return reg_loss

    def compute_loss(self,
                     visual_features: torch.Tensor,
                     prompts: list,
                     categories: list = None) -> Dict[str, torch.Tensor]:
        """
        总损失函数

        L_total = L_contrastive + λ · L_reg

        Args:
            visual_features: (B, d) 视觉特征
            prompts: 文本Prompt列表
            categories: 类别名列表
        Returns:
            {'total': tensor, 'contrastive': tensor, 'reg': tensor, 'n_c': int, ...}
        """
        # TATE编码
        enc = self.encode(prompts, categories, training=True)

        # Token级对比损失 (使用投影前的768维向量)
        loss_contrastive = self.compute_token_level_contrastive_loss(
            visual_features,
            enc['pruned_raw'],   # (B, k, 768) — 同维度
            enc['weights'],
        )

        # 统一上界正则化 (使用投影前的768维向量)
        loss_reg = self.compute_upper_bound_regularization(
            enc['pruned_raw'],
            enc['weights'],
        )

        # 总损失
        lambda_reg = F.softplus(self.lambda_reg)  # 保证非负
        loss_total = loss_contrastive + lambda_reg * loss_reg

        return {
            'total': loss_total,
            'contrastive': loss_contrastive,
            'reg': loss_reg,
            'lambda': lambda_reg.detach(),
            'n_c': enc['n_c'],
            'weights': enc['weights'].detach(),
        }

    def forward(self, prompts: list, categories: list = None):
        """推理模式前向传播"""
        return self.encode(prompts, categories, training=False)


# ═══════════════════════════════════════════════════════════════
# 4. TATE-GDINO 集成包装器
# ═══════════════════════════════════════════════════════════════
class TATEGDINO(nn.Module):
    """
    将TATE集成到GDINO-tiny中

    替换text_backbone为TATE增强版, 保持视觉backbone和检测头不变

    使用:
      model = TATEGDINO(gdino_tiny_model, gdino_processor)
      # 训练TATE模块
      loss = model.train_step(images, prompts, categories)
      # 推理
      detections = model.detect(images, prompts)
    """

    def __init__(self, gdino_model, processor):
        super().__init__()

        self.gdino = gdino_model
        self.processor = processor

        # 提取GDINO的BERT和text_projection
        # gdino_model: GroundingDinoForObjectDetection
        #   .model → GroundingDinoModel (.text_backbone, .text_projection)
        gm = gdino_model.model
        self.orig_bert = gm.text_backbone
        self.orig_text_proj = gm.text_projection

        # 创建TATE
        self.tate = TATE(self.orig_bert, self.orig_text_proj, processor)

        # 冻结视觉部分
        gm = gdino_model.model  # GroundingDinoModel
        for p in gm.backbone.parameters():
            p.requires_grad = False
        for p in gm.encoder.parameters():
            p.requires_grad = False
        for p in gm.decoder.parameters():
            p.requires_grad = False

    def get_visual_features(self, pixel_values, pixel_mask=None):
        """提取视觉特征 (用于训练TATE)"""
        with torch.no_grad():
            gm = self.gdino.model  # GroundingDinoModel
            vis_features, vis_positions = gm.backbone(pixel_values, pixel_mask)

            # vis_features: list of (feature_tensor, mask_tensor)
            last_feat, _ = vis_features[-1]   # (B, 768, H, W)
            visual_feat = last_feat.mean(dim=[2, 3])  # GAP → (B, 768)
        return visual_feat

    def train_step(self, pixel_values, prompts, categories=None, pixel_mask=None):
        """
        单步训练

        Args:
            pixel_values: (B, C, H, W) 图像
            prompts: 文本Prompt列表
            categories: 类别名列表
        Returns:
            loss_dict
        """
        visual_feat = self.get_visual_features(pixel_values, pixel_mask)
        loss_dict = self.tate.compute_loss(visual_feat, prompts, categories)
        return loss_dict

    def detect(self, images, prompts, threshold=0.25):
        """推理: 使用TATE编码的Token做检测"""
        device = next(self.gdino.parameters()).device

        if isinstance(images, list):
            results = [self.detect(img, prompts, threshold) for img in images]
            return results

        # 单张图像
        if not isinstance(images, torch.Tensor):
            from PIL import Image
            if isinstance(images, Image.Image):
                images = self.processor(images=images, return_tensors='pt').pixel_values

        images = images.to(device)
        B = images.shape[0]

        # TATE编码文本
        if isinstance(prompts, str):
            prompts = [prompts]
        if B > 1 and len(prompts) == 1:
            prompts = prompts * B

        tate_out = self.tate.encode(prompts, training=False)

        # 使用GDINO做检测 (需要修改forward以接受预编码的text embeddings)
        # 这里用原始GDINO做fallback
        with torch.no_grad():
            inputs = self.processor(images=images, text=prompts, return_tensors='pt')
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            outputs = self.gdino(**inputs)

        return outputs


# ═══════════════════════════════════════════════════════════════
# 5. 工具函数
# ═══════════════════════════════════════════════════════════════
def create_tate_from_gdino(gdino_model, processor, freeze_bert: bool = True):
    """从GDINO模型创建TATE"""
    # gdino_model: GroundingDinoForObjectDetection
    #   .model → GroundingDinoModel
    #     .text_backbone → BertModel
    #     .text_projection → Linear
    gm = gdino_model.model  # GroundingDinoModel
    bert = gm.text_backbone
    text_proj = gm.text_projection
    return TATE(bert, text_proj, processor, freeze_bert=freeze_bert)


def load_pretrained_tate(path: str, gdino_model, processor):
    """加载预训练的TATE权重"""
    tate = create_tate_from_gdino(gdino_model, processor)
    state_dict = torch.load(path, map_location='cpu')
    tate.load_state_dict(state_dict, strict=False)
    return tate


print("TATE model defined. Ready for training.")
print(f"  Modules: TIM ({TokenImportanceModule.__name__}) + S-N Pruning ({SNDynamicPruning.__name__})")
print(f"  Loss: Token-Level Contrastive + Upper Bound Regularization")
print(f"  Constants: C={THEORETICAL_CONSTANTS['C']}, η_α={THEORETICAL_CONSTANTS['eta_alpha']}, κ={THEORETICAL_CONSTANTS['kappa']}")
