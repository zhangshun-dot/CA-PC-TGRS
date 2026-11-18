# 跨域 Token 分布迁移理论

## 一、符号约定

| 符号 | 含义 |
|------|------|
| S, T | 源域（遥感）、目标域（自然图像） |
| π_D | 域 D 的最优前缀文本（如 "an overhead image of"） |
| c | 目标类别名称 |
| m_a | 修饰轴 a 对应的英文短语 |
| δ_D(c, a) | 域 D 中，类别 c 在修饰轴 a 上的 DR 增益：DR(π_D + c + m_a) − DR(π_D + c) |
| M*_D(c) | 域 D 中类别 c 的最优修饰词集合：{a : δ_D(c, a) > τ} |
| f_enc | 文本编码器（BERT/CLIP）：prompt → token embeddings |

## 二、已有定理（复用 TDM 论文）

**定理 1（轴偏好 Lipschitz 连续性）**：对于同类文本编码器下的任意两个 token 嵌入向量 e₁, e₂：

$$|\delta(c, a|e_1) - \delta(c, a|e_2)| \leq L_a \cdot \|e_1 - e_2\|$$

其中 $L_a < 1$（当前论文已验证：Banach 不动点 + PCA 前 7 主成分 84.2% 方差 → 收缩映射）。

## 三、核心假设：域-修饰轴近似正交

**假设 1（域前缀与修饰轴子空间近似正交）**：域前缀 π 的影响主要集中在 baseline DR 上，对修饰轴增益 δ 的影响远小于同类语义差异。即，对同一类别 c，换域前缀主要平移基线，而不改变各轴的 delta 符号：

$$\text{sign}(\delta_S(c, a)) \approx \text{sign}(\delta_T(c, a)) \quad \text{对大多数 } c, a$$

**已有预实验支撑**：
- DIOR A/B 子集 Pearson r = 0.77（同域内部子集分裂）
- DIOR → DOTA Pearson r = 0.9253（跨数据集，同遥感域）
- 这两组数据表明：**修饰轴增益的排序在不同数据划分下高度稳定**

## 四、跨域迁移上界定理

**定理 2（跨域 Token 迁移上界）**：设 $M^*_S(c)$ 为源域 S 的 OFAT 最优修饰集，$\pi_T$ 为目标域 T 经前缀标定后的最优前缀。则在目标域 T 上，跨域迁移方案与在 T 上重新做完整 OFAT 之间的性能差满足：

$$\Delta_{\text{cross}}(c) \equiv DR_T(\pi_T + c + M^*_T) - DR_T(\pi_T + c + M^*_S) \leq \sum_{a \in M^*_T \Delta M^*_S} |\delta_T(c, a)|$$

其中 $M^*_T \Delta M^*_S$ 为对称差（只在某一域被选中但另一域未选中的轴）。

**进一步上界**：由定理 1 的 Lipschitz 连续性：

$$|\delta_S(c, a) - \delta_T(c, a)| \leq L_a \cdot \|\pi_S - \pi_T\| + \varepsilon_{\text{cross}}$$

其中 $\varepsilon_{\text{cross}}$ 为残差跨域效应。

当 $|\delta_S(c, a) - \delta_T(c, a)| < \tau$（即轴偏好跨域偏移小于筛选阈值），两个域选出的修饰词集相同（$M^*_S = M^*_T$），此时 $\Delta_{\text{cross}} = 0$。

当偏移超过 $\tau$，最差情况下：

$$\Delta_{\text{cross}}(c) \leq |M_{\max}| \cdot \max_a (L_a \cdot \|\pi_S - \pi_T\| + \varepsilon_{\text{cross}} + \tau)$$

由 S-N 约束，$|M_{\max}| \leq 2$，且 $L_a < 1$，故：

$$\boxed{\Delta_{\text{cross}}(c) \leq 2 \cdot (\|\pi_S - \pi_T\| + \varepsilon_{\text{cross}} + \tau)}$$

## 五、关键推论

**推论 1（前缀标定消去主项）**：跨域迁移只需做**域前缀重新标定**（在 T 上测试 5 个候选前缀），即可消去 $\|\pi_S - \pi_T\|$ 项，使上界退化为：

$$\Delta_{\text{cross}}^{\text{calibrated}}(c) \leq 2 \cdot (\varepsilon_{\text{cross}} + \tau)$$

**推论 2（跨域可迁移性条件）**：当 $\varepsilon_{\text{cross}} < \tau$（残差跨域效应小于筛选阈值），仅前缀标定即可实现无损跨域迁移，即：

$$M^*_S(c) = M^*_T(c) \quad \text{对大多数 } c$$

**推论 3（跨模型-跨域对称性）**：TDM 论文已证明跨模型迁移只需前缀标定（5 次比较）。本文将此对称地推广到跨域情形，形成统一的"跨 X 迁移"框架：

| 迁移类型 | TDM 论文 | CD-TDM（本文） |
|----------|----------|----------------|
| 保持项 | 修饰轴增益（BERT OFAT） | 修饰轴增益（源域 OFAT） |
| 标定项 | 目标模型前缀（5 次） | 目标域前缀（5 次） |
| 数学保证 | 同一编码器 δ 不变 | 域前缀与修饰轴近似正交 |

## 六、待实验验证

上述推导给出的是**理论上界**。核心实验验证：

1. **COCO 上直接复用 DIOR OFAT**（不做前缀标定）→ 验证上界紧致性
2. **前缀标定后复用**（5 次 COCO 前缀测试 + DIOR 修饰词方案）→ 验证推论 1
3. **控制对照**：COCO 上重做完整 OFAT → 金标准对比
4. **预期结果**：标定后 $\Delta_{\text{cross}}$ 的小于 10%（推论 1 预测），证明 $\varepsilon_{\text{cross}}$ 在可控范围内

---

## 七、与 S-N 上界的整合

将跨域迁移上界与 S-N 统一上界整合，得到**跨域 OVD Prompt 优化的完整信息上界**：

$$I_{\text{eff}}^{\text{cross}} \leq C \cdot \left(1 - e^{-n / n_c}\right) - \gamma \cdot D_{\text{KL}}(P_S \| P_T)$$

其中 $\gamma$ 为域间互信息衰减因子，$D_{\text{KL}}(P_S \| P_T)$ 为两个域的 prompt token 分布之间的 KL 散度。

当 $D_{\text{KL}}(P_S \| P_T)$ 很小（域间 token 分布接近），或前缀标定已消除大部分域差异时，跨域迁移几乎无损。
