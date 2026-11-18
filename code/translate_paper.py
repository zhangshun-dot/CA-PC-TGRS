"""
Translate A-CSPC paper from Chinese to English
Following nature-polishing guidelines:
- Extract core propositions first
- Reconstruct logical links
- Verify terminology and hedging
- Keep technical terms stable
"""
from docx import Document
from docx.shared import Pt
from copy import deepcopy

# Load original document
doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# Translation dictionary: paragraph_index -> English text
# We'll build this incrementally
translations = {}

# ============================================================
# TITLE (para 0)
# ============================================================
translations[0] = (
    "Asymmetric Cross-Space Prototype Calibration for Cross-Domain "
    "Object Detection: Failure Diagnosis and Repairable Boundary"
)

# ============================================================
# ABSTRACT (paras 1-4)
# ============================================================
translations[1] = "Abstract"

translations[2] = (
    "In cross-domain object detection, source-domain visual prototypes "
    "systematically fail under shifts in both imaging domain and category "
    "granularity. We propose A-CSPC, an Asymmetric Cross-Space Prototype "
    "Calibration method positioned as a proposal re-ranking corrector for "
    "weakly-calibrated open-vocabulary detectors. Positive prototypes are "
    "extrapolated along the CLIP text direction, while negative prototypes "
    "are retained in the source domain; detector scores and visual prototype "
    "scores are fused by geometric mean over a fixed proposal pool. On the "
    "DIOR-to-COCO six-class benchmark, the target-domain-tuned configuration "
    "(beta=2.0, lambda=0.5) achieves +0.0646 AP50 and +0.0463 AP@[0.5:0.95] "
    "(6/6 classes positive), and is replicated across three alternative source "
    "domains (NWPU, DOTA, VisDrone). The target-domain-agnostic configuration "
    "(1.0, 0.25) yields +0.0166 with boat turning negative."
)

translations[3] = (
    "This work is positioned as an analytical study of cross-domain prototype "
    "failure diagnosis and repairable boundaries. Layered failure diagnosis "
    "reveals that the ranking improvement from extrapolation (Delta AUC), "
    "rather than the degree of source prototype failure, is predictive "
    "(rho=+0.577). A class-level physical prior framework is proposed, "
    "demonstrating ex ante per-class gain prediction (Spearman rho=+0.699 "
    "across 37 classes and 6 domain pairs, though domain-pair-clustered "
    "bootstrap two-sided 95% CI contains zero, constituting exploratory "
    "evidence). Under a given risk score, the resulting target-domain-annotation-free "
    "adaptive parameter control achieves additional gain over fixed parameters "
    "(+0.0461 vs. +0.0166), validating the control mechanism itself. The "
    "asymmetric construction of A-CSPC achieves +0.0646 on DIOR-to-COCO under "
    "target-domain tuning, and the repairable boundary is systematically "
    "delineated through cross-source replication, reverse component attribution, "
    "and weak-to-strong detector spectrum experiments (applicable to weakly "
    "calibrated detectors; in the reverse fine-to-coarse direction, fine-grained "
    "prototypes are already highly clustered around the coarse centroid, requiring "
    "no correction)."
)

translations[4] = (
    "Keywords: cross-domain object detection; prototype calibration; failure "
    "diagnosis; vision-language model; training-free; remote sensing imagery"
)

# ============================================================
# INTRODUCTION (paras 5-15)
# ============================================================
translations[5] = "1. Introduction"

translations[6] = (
    "Cross-domain object detection must simultaneously handle shifts in imaging "
    "domain and category granularity. Existing methods based on source-domain "
    "visual prototypes typically assume that source prototypes can be directly "
    "transferred to the target domain [1-3], but this assumption is unreliable "
    "in the transfer from remote sensing overhead views to natural scene images. "
    "For example, the coarse-grained vehicle prototype from DIOR [4], when "
    "transferred to bus and motorcycle in COCO [5], yields ranking AUCs of only "
    "approximately 0.20 and 0.29 within the proposal pool, below random level. "
    "This work focuses on the setting where the detector is specified by text "
    "categories and requires no target-domain training; the core problem is "
    "cross-domain visual prototype calibration and re-scoring."
)

translations[7] = (
    "We focus on the inference-stage setting requiring no training or target-domain "
    "adaptation: the source domain has annotations used to build visual prototypes; "
    "the method performs no training or test-time adaptation at inference, only "
    "recomputing and fusing category scores over a fixed proposal pool. Four "
    "information levels must be clearly distinguished: (i) the main A-CSPC "
    "configuration uses no target-domain images or annotations at inference; "
    "(ii) the main configuration (beta=2.0, lambda=0.5) uses target-domain GT "
    "for hyperparameter tuning during development (disclosed in Section 4.9), "
    "and does not claim strict zero-shot; (iii) the development-domain-selected "
    "configuration (1.0, 0.25) is tuned on DIOR-to-VOC and does not contact the "
    "target domain at test time; (iv) realized gamma per-class control uses "
    "target-domain baseline AP (requiring GT), constituting oracle upper-bound "
    "analysis."
)

translations[8] = (
    "In the DIOR-to-COCO transfer, we observe that source-domain visual prototypes "
    "do not merely degrade in performance but systematically fail on specific "
    "categories. Taking the coarse-grained vehicle prototype from DIOR transferred "
    "to bus and motorcycle in COCO as an example, the ranking AUCs within the "
    "proposal pool are approximately 0.20 and 0.29, below the random ranking "
    "level of 0.5; correspondingly, the source-domain visual prototype baseline "
    "(ablation configuration removing extrapolation) drops by 0.0710 on average "
    "across six classes, with bus and motorcycle dropping by approximately 0.294 "
    "and 0.250, respectively. This indicates that the key problem in cross-domain "
    "detection is not that all categories require the same strength of adaptation, "
    "but rather that one must first determine whether the source prototype has "
    "failed, and then decide whether to extrapolate. Based on this observation, "
    "this work organizes the problem into three pillars: diagnosis, repair, and "
    "boundary."
)

translations[9] = "The main contributions of this work are as follows:"

translations[10] = (
    "(1) We propose A-CSPC, an Asymmetric Cross-Space Prototype Calibration method, "
    "positioned as an analytical vehicle for cross-domain prototype failure diagnosis "
    "and repairable boundaries. The method is organized around diagnosis-repair-boundary: "
    "layered failure diagnosis reveals that the ranking improvement from extrapolation "
    "(Delta AUC), rather than the degree of source prototype failure, is predictive "
    "(rho=+0.577 across 32 classes and 5 target domains); the asymmetric construction "
    "of positive prototype extrapolation along the CLIP text direction and negative "
    "prototype retention in the source domain achieves +0.0646 gain under the "
    "target-domain-tuned configuration, and the necessity of the asymmetric construction "
    "is validated through minimal baselines."
)

translations[11] = (
    "(2) We propose a class-level physical prior framework, implementing a complete "
    "pipeline from ex ante prediction to adaptive control. A remote sensing ground "
    "object prior library is constructed, and after factor ablation on the development "
    "set, four factors (G/S/I/C: granularity mismatch, scale/resolution, intra-class "
    "variance, inter-class confusion) are finally adopted (V/M are noise factors). "
    "Spearman rho=+0.699 is achieved across 37 classes and 6 domain pairs, but "
    "domain-pair-clustered bootstrap shows that the two-sided 95% CI contains zero "
    "(one-sided rho>0 proportion 95.6%, with limited K=6); this predictive power "
    "should be regarded as exploratory evidence requiring confirmation from more "
    "independent domain pairs. GT-assisted per-class control (realized gamma) based "
    "on this prediction serves as oracle upper-bound analysis."
)

translations[12] = (
    "(3) We systematically delineate the repairable boundary and information usage "
    "protocol of the method. Through cross-source replication across three source "
    "domains, reverse component attribution (text extrapolation direction mismatch "
    "as the primary cause of reverse failure), and weak-to-strong detector spectrum "
    "experiments, we validate the robustness of the method to the source prototype "
    "library and clarify its corrector boundary (gain approaches zero or even turns "
    "negative on strong detectors, -0.0316 on OWLv2, clarifying that the method is "
    "only applicable to weakly calibrated detectors); we propose a hyperparameter "
    "configuration audit protocol, clarifying the target-domain tuning source of "
    "beta=2.0/lambda=0.5, and clearly distinguishing the target-domain-tuned "
    "configuration (+0.0646) from the target-domain-agnostic configuration (+0.0166, "
    "boat turning negative)."
)

translations[13] = (
    "Essential differences from existing prototype methods: CoOp [10] (learning "
    "optimizable text prompt vectors) performs prototype learning only in text space, "
    "without cross-space extrapolation of visual prototypes; TGE (text-guided "
    "extrapolation) adopts symmetric extrapolation for both positive and negative "
    "prototypes, without distinguishing their different transfer characteristics. "
    "The core increment of this work is the asymmetric construction: positive "
    "prototypes are extrapolated along the CLIP text direction to the target domain "
    "to adapt to imaging domain shifts, while negative prototypes are retained in the "
    "source domain to avoid background noise amplification. This asymmetric design is "
    "validated as necessary in ablation experiments (the symmetric extrapolation EC "
    "configuration drops from +0.0652 to +0.0034)."
)

translations[14] = (
    "In response to questions about the value of the method in the zero-shot setting, "
    "we further propose physical-prior-driven adaptive parameter control without "
    "target-domain annotations: using class-level physical priors (granularity mismatch, "
    "scale, intra-class variance, inter-class confusion) to predict per-class correction "
    "needs, directly driving adaptive hyperparameters beta_eff(c)=beta_base*(1+alpha*R(c)), "
    "achieving +0.0461 gain on DIOR-to-COCO six classes (fixed parameters only +0.0166, "
    "relative improvement 177%), with 6/6 classes positive, and validating generality in "
    "the VOC-to-COCO 20-class extension (adaptive 18/20 classes positive, 15/20 classes "
    "better than fixed parameters). This mechanism upgrades the main configuration from "
    "\"target-domain tuning\" to \"physical-prior-driven annotation-free adaptive control,\" "
    "directly addressing the question of method value in the zero-shot setting."
)

translations[15] = (
    "Scope and limitations of this work's claims: (1) This work is positioned as an "
    "analytical study of cross-domain prototype failure diagnosis and repairable boundaries, "
    "and A-CSPC is an analytical vehicle rather than a performance SOTA method; (2) the "
    "main result (beta=2.0, lambda=0.5) relies on target-domain GT tuning, and the "
    "target-domain-agnostic configuration (1.0, 0.25) yields only +0.0166 with boat "
    "turning negative; (3) the main experimental target classes belong to categories seen "
    "during detector pretraining, and open-vocabulary nature is embodied in \"categories "
    "specified by text, requiring no target-domain training\"; (4) the physical prior in "
    "Section 6 is exploratory research, with full-data rho=+0.699 but cluster bootstrap "
    "two-sided CI containing zero, with limited K=6; predictive power requires confirmation "
    "from more independent domain pairs and currently does not constitute a deployable "
    "component."
)

# ============================================================
# Apply translations to document
# ============================================================
def apply_translations(doc, translations):
    """Apply translations to document paragraphs, preserving style"""
    for idx, text in translations.items():
        if idx < len(doc.paragraphs):
            p = doc.paragraphs[idx]
            # Clear existing runs
            for run in p.runs:
                run.text = ""
            # Add new text to first run or create new run
            if p.runs:
                p.runs[0].text = text
            else:
                p.add_run(text)
    return doc

# Apply translations
doc = apply_translations(doc, translations)

# Save translated document
output_path = r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v1.docx'
doc.save(output_path)
print(f"Translated document saved to: {output_path}")
print(f"Translated {len(translations)} paragraphs (abstract + intro)")
print()
print("Next steps:")
print("  - Translate related work (Section 2)")
print("  - Translate methods (Section 3)")
print("  - Translate experiments (Section 4)")
print("  - Translate physical prior (Section 5-6)")
print("  - Translate limitations and conclusion (Section 7-8)")
print("  - Translate table captions")
