"""
Translate Sections 2 (Related Work) and 3 (Methods)
Continue from previous translation
"""
from docx import Document

# Load the partially translated document
doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v1.docx')

# Additional translations for Sections 2 and 3
translations = {}

# ============================================================
# SECTION 2: RELATED WORK (paras 16-25)
# ============================================================
translations[16] = "2. Related Work"

translations[17] = "2.1 Cross-Domain Object Detection"

translations[18] = (
    "Cross-domain object detection aims to transfer a detector trained on the source "
    "domain to a target domain with different imaging conditions, viewpoints, or category "
    "distributions. Existing methods are mainly divided into three categories: domain "
    "adversarial training methods [6-8] reduce inter-domain differences through feature "
    "alignment; self-training methods [9] iteratively optimize using target-domain pseudo "
    "labels; prototype matching methods [10] establish category prototypes in feature space "
    "and classify by nearest neighbor. This work belongs to the prototype matching category, "
    "but focuses on the systematic failure of source-domain prototypes in cross-domain "
    "transfer, rather than assuming prototypes can be directly transferred."
)

translations[19] = "2.2 Open-Vocabulary Detection and Prototype Methods"

translations[20] = (
    "Text-specified cross-domain proposal re-ranking (a subtask of open-vocabulary detection) "
    "uses vision-language models (such as CLIP [1]) to map text category descriptions to visual "
    "feature space, enabling detection of categories unseen during training. Representative methods "
    "include GLIP [11], Grounding DINO [2], OWLv2 [3], and YOLO-World [12]. These methods "
    "typically match proposal features with text embeddings, but in cross-domain scenarios, the "
    "text-visual alignment learned in the source domain may fail. Prototype methods [10] establish "
    "prototypes by aggregating source-domain category features and then perform nearest-neighbor "
    "matching in the target domain. Methods such as TGE attempt to improve cross-domain matching "
    "through text-guided prototype extrapolation, but do not systematically diagnose the causes of "
    "prototype failure, nor distinguish the different transfer characteristics of positive and "
    "negative prototypes."
)

translations[21] = (
    "The proposed A-CSPC introduces asymmetric cross-space calibration on top of source-domain "
    "prototypes: positive prototypes are extrapolated along the CLIP text direction to the target "
    "domain, while negative prototypes are retained in the source domain to suppress inter-class "
    "confusion."
)

translations[22] = "2.3 Prompt Learning, Feature Adaptation, and Test-Time Adaptation"

translations[23] = (
    "Prompt learning (such as CoOp [10], CoCoOp [21]) improves the zero-shot classification "
    "performance of CLIP by learning optimizable text prompt vectors, but requires source-domain "
    "training data and the learnable parameters may overfit. Feature adaptation methods (such as "
    "CLIP-Adapter [31], Tip-Adapter [16]) add lightweight adaptation layers on frozen CLIP features, "
    "also requiring training data. Test-time adaptation (TTA) methods [17-20] perform online "
    "optimization using unlabeled target-domain data at inference, but suffer from catastrophic "
    "forgetting and instability. The proposed A-CSPC is a training-free method: no training or "
    "adaptation is performed at inference, only recomputing and fusing category scores over a fixed "
    "proposal pool. The information budget differs from CoOp/CLIP-Adapter: the latter uses source-domain "
    "annotations to train learnable parameters, while the main A-CSPC configuration uses target-domain "
    "annotations for hyperparameter selection (disclosed in Section 4.9)."
)

translations[24] = "2.4 Detector Proposal Quality and Repairable Space"

translations[25] = (
    "Proposal re-ranking methods [21,22] use additional classifiers or matching scores to re-rank "
    "proposals output by the detector, improving detection accuracy without changing proposal recall. "
    "The upper bound of gain for such methods is determined by the maximum achievable recall of the "
    "proposal pool: if the true target is not in the proposal pool, no re-ranking can recover it. "
    "The proposed A-CSPC belongs to the proposal re-ranking category, and its corrector boundary is "
    "determined by the ranking calibration state of the detector baseline: weak detectors (such as "
    "GDINO-tiny) have large ranking noise and large re-ranking gain space; strong detectors (such as "
    "OWLv2) have approximately saturated ranking, with gain approaching zero or even turning negative "
    "(Section 4.8)."
)

# ============================================================
# SECTION 3: METHODS (paras 26-40)
# ============================================================
translations[26] = "3. Problem Definition and Method"

translations[28] = (
    "Figure 1. Overall pipeline of A-CSPC for cross-domain object detection (three pillars of "
    "diagnosis-repair-boundary: P1 layered failure diagnosis distinguishes ranking quality from "
    "proposal coverage; P2 asymmetric prototype construction extrapolates positive prototypes along "
    "the text direction while retaining source-domain negative prototypes; P3 direction gate g2 "
    "selects between extrapolation and conservative mode)."
)

translations[29] = "3.1 Cross-Domain Detection Setting (Categories Specified by Text)"

translations[30] = (
    "Given a source domain S (e.g., remote sensing imagery DIOR) and a target domain T (e.g., natural "
    "scene imagery COCO), both share the category set C but differ in imaging domain and category "
    "granularity. The detector D, with categories specified by text, generates a proposal set B={b_k} "
    "and detector scores s_det(b_k) for the target-domain image I. The goal of A-CSPC is to re-rank "
    "proposals using source-domain visual prototypes without retraining the detector. The main "
    "experimental target classes belong to categories seen during detector pretraining, and the "
    "open-vocabulary nature is embodied in \"categories specified by text, requiring no target-domain "
    "training.\""
)

translations[31] = "3.2 P1: Source Prototype Failure Diagnosis"

translations[33] = (
    "Figure 2. Layered failure diagnosis: ranking AUC and proposal pool recall must be reported "
    "separately (P1 diagnosis distinguishes ranking quality from proposal coverage; 6-class average "
    "ranking AUC 0.576, proposal pool GT coverage rate 0.743)."
)

translations[34] = (
    "The source-domain visual prototype p_c is obtained by aggregating annotated box features of "
    "category c in the source domain. In cross-domain transfer, the ranking quality of p_c within the "
    "target-domain proposal pool is measured by AUC: using IoU >= 0.5 between proposals and ground-truth "
    "boxes as positive labels, the ranking AUC of the cosine similarity between p_c and proposal features "
    "is computed. Layered failure diagnosis distinguishes two dimensions: (1) proposal coverage rate R_max: "
    "the proportion of true targets contained in the proposal pool, determining the upper bound of "
    "re-ranking gain; (2) proposal ranking quality AUC: the discriminative ability of the source prototype "
    "for proposals. Key finding: what is truly predictive is the ranking improvement from extrapolation, "
    "Delta AUC = AUC_extrapolated - AUC_source_prototype, rather than the degree of source prototype failure "
    "or the original AUC (rho=+0.577 across 32 classes and 5 target domains)."
)

translations[35] = "3.3 P2: Asymmetric Cross-Space Prototype Construction"

translations[36] = (
    "The core of A-CSPC is the asymmetric construction: positive prototypes are extrapolated along the "
    "CLIP text-domain offset direction to the target domain space, while negative prototypes are retained "
    "in the source domain space. This design is based on the observation that positive prototypes need to "
    "adapt to the imaging characteristics of the target domain, while negative prototypes (inter-class "
    "confusion categories) can already effectively suppress background in the source domain space, and "
    "extrapolation反而 introduces noise. The positive prototype extrapolation formula is shown in Eq. (1): "
    "p_hat_c = p_c + beta * (d(x_c) - s(x_s(c)))  (Eq. 1), where d(x_c) is the CLIP embedding of the "
    "target-domain text template (\"a photo of {x_c}\"), s(x_s(c)) is the CLIP embedding of the source-domain "
    "text template (\"an overhead image of {x_s(c)}\"), x_s(c) is the mapped category of category c in the "
    "source domain, and beta is the extrapolation strength coefficient; the extrapolated prototype is L2-normalized "
    "(since only cosine similarity is used subsequently, normalization does not change cosine values). The "
    "negative prototype p_neg is aggregated from the top-k source-domain classes closest in cosine to the "
    "positive prototype, without extrapolation."
)

translations[37] = (
    "The final fusion score uses geometric mean. The visual prototype score is defined as shown in Eq. (2): "
    "s_vis(f_k) = max(0, <f_k, p_hat_c> - lambda * <f_k, p_neg>)  (Eq. 2), where <.,.> denotes cosine "
    "similarity and lambda is the negative prototype suppression coefficient. ReLU truncation (max(0,.)) "
    "ensures non-negative visual scores. The final fusion score is shown in Eq. (3): "
    "s_fused,k = sqrt(clip(s_det,k, epsilon, 1) * clip(s_vis(f_k), epsilon, 1)), epsilon=10^-6  (Eq. 3), "
    "where clip(z, epsilon, 1) = min(1, max(epsilon, z)). To ensure the geometric mean is always defined, "
    "clip(epsilon,1) must be applied separately to the detector score s_det and the visual score s_vis, "
    "followed by square root, rather than only truncating the final result. The geometric mean ensures that "
    "high scores are achieved only when both the detector score and the prototype score are non-zero, avoiding "
    "dominance by a single score."
)

translations[38] = "3.4 P3: Direction-Aware Selective Extrapolation"

translations[39] = (
    "Direction-aware selective extrapolation controls the extrapolation strength through the domain direction "
    "gate g2: when g2 > 0, extrapolation is performed along the source-to-target text direction; when g2 <= 0, "
    "it falls back to pure source-domain prototypes. g2 is jointly determined by the text offset direction of the "
    "domain pair and the category granularity mismatch. Reverse transfer experiments (COCO-to-NWPU) show that "
    "the text extrapolation direction has clear direction selectivity: photo-to-overhead direction extrapolation "
    "leads to performance degradation (component attribution in Section 5.7), so A-CSPC is not applicable to "
    "arbitrary domain pairs, and its applicability boundary is determined by the validity of the text offset "
    "direction."
)

translations[40] = (
    "First, the global domain direction is constructed using the category directions between source-domain and "
    "target-domain text templates: D_dom = mean_{y in Y}(d(y) - s(x_s(y))), where Y is the category set "
    "participating in construction, and x_s(y) is the source-domain mapping of category y. Then the direction "
    "consistency score for each category is computed: g2(c) = cos(d(x_c) - s(x_s(c)), D_dom) * "
    "(1 - |granularity(c) - granularity_ref|), where d(x_c) is the target-domain text template embedding and "
    "s(x_s(c)) is the source-domain text template embedding. When g2 > 0, extrapolation is performed along the "
    "source-to-target direction; when g2 <= 0, it falls back to pure source-domain prototypes."
)

# ============================================================
# Apply translations
# ============================================================
def apply_translations(doc, translations):
    for idx, text in translations.items():
        if idx < len(doc.paragraphs):
            p = doc.paragraphs[idx]
            for run in p.runs:
                run.text = ""
            if p.runs:
                p.runs[0].text = text
            else:
                p.add_run(text)
    return doc

doc = apply_translations(doc, translations)

# Save
output_path = r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2.docx'
doc.save(output_path)
print(f"Translated document saved to: {output_path}")
print(f"Added {len(translations)} new translations (Sections 2-3)")
print(f"Total translated: {16 + len(translations)} paragraphs")
