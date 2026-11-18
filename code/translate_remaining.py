"""
Translate remaining Chinese paragraphs:
1. Fix paragraphs with mixed Chinese
2. Translate supplementary material section (214-238)
"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_final.docx')

translations = {}

# ============================================================
# FIX MIXED CHINESE PARAGRAPHS
# ============================================================
translations[36] = (
    "The core of A-CSPC is the asymmetric construction: positive prototypes are extrapolated along the CLIP text-domain offset direction to the target domain space, while negative prototypes are retained in the source domain space. This design is based on the observation that positive prototypes need to adapt to the imaging characteristics of the target domain, while negative prototypes (inter-class confusion categories) can already effectively suppress background in the source domain space, and extrapolation instead introduces noise. The positive prototype extrapolation formula is shown in Eq. (1): p_hat_c = p_c + beta * (d(x_c) - s(x_s(c))) (Eq. 1), where d(x_c) is the CLIP embedding of the target-domain text template (\"a photo of {x_c}\"), s(x_s(c)) is the CLIP embedding of the source-domain text template (\"an overhead image of {x_s(c)}\"), x_s(c) is the mapped category of category c in the source domain, and beta is the extrapolation strength coefficient; the extrapolated prototype is L2-normalized (since only cosine similarity is used subsequently, normalization does not change cosine values). The negative prototype p_neg is aggregated from the top-k source-domain classes closest in cosine to the positive prototype, without extrapolation."
)

translations[52] = (
    "DANN training instability analysis: to exclude the质疑 that \"domain adversarial failure is due to insufficient training data or single-run deviation\", two additional experiments were conducted: (1) target-domain training image count scanned from 10 to 200 (10/50/100/200 images), 6-category mean gains are -0.0958, -0.0087, -0.0049, +0.0009 (single run); (2) 5 random seeds for the 200-image configuration, 6-category mean gain is -0.0551 +/- 0.1090. Results show that DANN training is extremely unstable under this setting: some seeds learn near-identity transformations (AP identical to baseline), some seeds completely collapse (e.g., airplane category seed=42 gain -0.4333). The multi-seed mean is negative (-0.0551), but the standard deviation reaches 0.1090 (2.0 times the absolute value of the mean), 95% confidence interval [-0.1904, +0.0802] crosses zero, so it cannot be判定 as significantly harming performance; the robust conclusion is that domain adversarial training under this setting not only has no gain, but also does not exceed CORAL (+0.0210) and A-CSPC physical prior adaptive under any of the 5 seeds..."
)

# Fix "质疑" -> "question", "判定" -> "determined"
translations[52] = translations[52].replace("质疑", "question").replace("判定", "determined")

translations[58] = (
    "Results show that A-CSPC achieves stable gain on VOC-to-COCO 20 categories: fixed-parameter configuration 20-category average +0.0301 (19/20 categories positive), physical prior adaptive configuration 20-category average +0.0328 (18/20 categories positive). Adaptive prior has further improvement compared to fixed parameters (+0.0328 vs. +0.0301), validating the effectiveness of physical-prior-driven per-class parameter control on broader categories. The two negative-gain categories (potted plant -0.0052, train -0.0048) both have magnitude less than 0.006,属于 acceptable boundary cases; among them, potted plant is -0.0298 under fixed parameters, and adaptive prior improves it to -0.0052, indicating that physical prior risk modulation has a repairing effect on difficult categories. This extended category validation indicates that the gain of A-CSPC does not depend on the special structure of DIOR-to-COCO 6 categories, and remains stable and effective on broader 20 categories and more diverse category types."
)
translations[58] = translations[58].replace("属于", "belonging to")

translations[69] = (
    "Additional ablation experiments: (a) pre-filtering threshold sensitivity: tested at thresholds 0.01/0.03/0.05 respectively, 6-category mean gains are +0.0615/+0.0567/+0.0517, all maintaining significant positive gain, indicating that the method is insensitive to the pre-filtering threshold and robust over a wide range. When the threshold decreases, proposals increase and baseline AP decreases, but the absolute gain of A-CSPC反而 increases, indicating that the method has larger correction space on low-quality proposal pools. (b) source-domain sample size sensitivity: using 5/10/20 annotated boxes per category from DIOR source domain to construct prototypes, 6-category mean gains are +0.0647/+0.0641/+0.0652, with differences among the three less than 0.001, indicating that the method has extremely low demand for source-domain annotation data, requiring only 5 samples per category to achieve gain comparable to 20 samples. This characteristic makes the method particularly suitable for cross-domain scenarios with scarce source-domain annotations."
)
translations[69] = translations[69].replace("反而", "instead")

translations[72] = (
    "Reverse COCO-to-DIOR transfer (3 DIOR target categories aggregated protocol, N=20 image protocol) is used to validate the applicability boundary of the A-CSPC extrapolation component. [Protocol note] Section 4.5 is cross-dataset overall reverse validation; Section 5.7 is a fixed COCO-to-NWPU, three-category, same CLIP encoding component control variable experiment. The two are not directly compared numerically. Results show a pattern opposite to the forward direction: conservative B2 (beta=0, pure source-domain visual prototype) averages +0.0418 with 3/3 categories positive, indicating that natural-scene source prototypes do not fail on the remote sensing target domain; the full extrapolation mode averages -0.0244 (3/3 categories negative), and pure text T4 drops even more (-0.0793). Under the COCO-to-DIOR six-row细分 protocol (Section 4.7), results show that negative prototypes have an amplifying effect under the wrong extrapolation direction; under the COCO-to-NWPU strict control variable protocol (Section 5.7), it is further shown that the text extrapolation direction is the main damage source, and negative prototypes further amplify this damage under the wrong direction."
)
translations[72] = translations[72].replace("细分", "fine-grained")

translations[76] = (
    "Granularity interpretation of direction selectivity. The above findings elevate \"direction selectivity\" to the more essential \"granularity mismatch boundary\": A-CSPC's text direction extrapolation produces value when correcting \"source-domain coarse-grained to target-domain fine-grained\" (centroid to circumference, with bridgeable semantic gaps); while in \"source-domain fine-grained to target-domain coarse-grained\" (circumference to centroid, fine-grained prototypes already clustered around the centroid), the premise of correction (existence of bridgeable gaps) does not hold, and extrapolation反而 introduces noise. This boundary is determined by the geometric structure of CLIP space, independent of the choice of specific extrapolation direction - multi-template direction calibration (reducing 13.7% damage), target-domain negative prototype replacement (reducing 27.3% damage), and visual centroid contraction (damage reduced to approximately 0) all cannot exceed B2 conservative +0.0418, and the three jointly validate the geometric conclusion that \"reverse transfer requires no correction.\""
)
translations[76] = translations[76].replace("反而", "instead")

translations[79] = (
    "Therefore, A-CSPC should be positioned as: a cross-domain prototype corrector for a specific granularity direction (coarse-to-fine); in the fine-to-coarse direction, the method automatically falls back to B2 conservative mode (g2 <= 0), and source-domain visual prototypes directly transfer as optimal. The domain direction discrimination of the direction gate g2 achieves 19/19 accuracy on 19 domain pairs, and class-level decision accuracy is 16/19 (the former judges domain direction, the latter judges whether specific categories should be extrapolated). (Reverse transfer protocol quick reference: -0.0244 is COCO-to-DIOR 3-category aggregated N=20 images / -0.0866 is six-row细分 protocol / -0.239 is COCO-to-NWPU three-category control variable experiment, the three have different protocols and cannot be directly compared)."
)
translations[79] = translations[79].replace("细分", "fine-grained")

translations[88] = (
    "Component attribution of reverse transfer (COCO-to-NWPU, 3-category control variable experiment): Configuration A (pure source-domain visual prototype, no extrapolation) average gain +0.100, indicating that source-domain visual prototype re-ranking is still effective in reverse tasks; Configuration B (A + negative prototype) average gain +0.0767, negative prototypes have slight side effects; Configuration C (A + positive prototype text extrapolation) average gain -0.081, performance明显 drops after adding text extrapolation; Configuration D (full A-CSPC) average gain -0.239, negative prototypes further amplify damage under the wrong extrapolation direction. Beta scan (beta in {0, 0.25, 0.5, 1.0, 1.5, 2.0}, lambda=0) shows that performance持续 drops after beta > 0, supporting the conclusion that \"photo-to-overhead direction is invalid.\""
)
translations[88] = translations[88].replace("明显", "significantly").replace("持续", "continuously")

translations[100] = "Table 13 (Section 4.5 COCO-to-DIOR protocol, N=20 images/category, 3-category aggregated) Reverse transfer six-row fine-grained protocol results (COCO-to-DIOR)"

translations[103] = (
    "Negative prototypes are retained in source domain space rather than extrapolated to the target domain, based on two observations: (1) negative prototypes represent inter-class confusion category features (top-k categories closest in cosine to positive prototypes in the source domain pool), and the statistical characteristics of inter-class confusion structures differ less between source and target domains than foreground categories, with source-domain-space negative prototypes already able to effectively suppress inter-class confusion; (2) extrapolating negative prototypes introduces target-domain background noise,反而 reducing discriminability for true targets. Ablation experiments validate: symmetric extrapolation (negative prototypes follow extrapolation) reduces mean AP under the same lambda (0.4377 to 0.4197, N=500), and removing negative prototypes shrinks total gain by about 60% (+0.0652 to +0.0239, N=100)."
)
translations[103] = translations[103].replace("反而", "instead")

translations[119] = (
    "Component attribution experiments of reverse transfer (COCO-to-NWPU) (Section 4.7) show that: visual prototype re-ranking has direction robustness (pure source-domain visual prototypes still achieve +0.100 average gain in reverse tasks), while text extrapolation has direction selectivity (gain drops to -0.081 after adding photo-to-overhead text extrapolation, and full A-CSPC further drops to -0.239). Beta scan shows that performance持续 drops after beta > 0, supporting the conclusion that the direction itself is not applicable. Therefore, A-CSPC should be positioned as: an asymmetric prototype corrector for specific domain directions (overhead-to-photo) and coarse-to-fine granularity mismatch, rather than a general extrapolation method symmetric to transfer direction. Reverse negative results serve as method boundary..."
)
translations[119] = translations[119].replace("持续", "continuously")

# ============================================================
# TRANSLATE REFERENCES HEADING
# ============================================================
translations[162] = "References"

# ============================================================
# TRANSLATE SUPPLEMENTARY MATERIAL (214-238)
# ============================================================
translations[214] = "Supplementary Material (published with main text at submission)"

translations[215] = "Configuration Selection Exploratory Analysis (Complete Results of Main Text Section 4.9)"

translations[216] = (
    "Main text Section 4.9 reports the configuration audit of the fixed configuration (2.0, 0.5). This subsection reports the complete results of the exploratory selection mechanism cited in the main text: under the leave-one-target-domain-out (LODO) protocol, the coefficient selectors based on auxiliary development domains (DCCS-U, TrainBest) do not stably outperform the fixed configuration (Supplementary Table S1); soft routing..."
)

translations[217] = "Supplementary Table S1. Leave-one-target-domain-out coefficient selection holdout gain (AP50 mean gain; DCCS-U selects parameters by robust utility, TrainBest by maximum training domain mean, Oracle is post-hoc upper bound)"

translations[218] = (
    "Supplementary Table S1 source: tge_dccs_lodo.json. DCCS-U approaches oracle (0.0441) in the 3-domain subset but is a post-hoc observation; after adding VisDrone (0.0268), it is lower than the fixed configuration Fixed(2.0, 0.5) (0.0302)."
)

translations[219] = "Supplementary Table S2. Complete Ablation and Efficiency (Main Text Section 4.4 Simplified, Complete Version Here)"

translations[220] = "Lambda scan (N=500 per-image AP50, source: tge_probe_ap_st.json lam_scan):"

translations[221] = (
    "Lambda=0.5 is the six-category common peak; lambda=1.0 degenerates bit by bit to the N=500 per-category baseline (mean 0.3728); lambda=0.75 shows a cliff-like drop."
)

translations[222] = "Beta scan (N=100 main table protocol, source: tge_probe_beta.json):"

translations[223] = (
    "Pre-filter sensitivity (N=100 main table protocol, source: tge_probe_prefilter.json): 0.01 / 0.03 / 0.05 -> gain +0.0699 / +0.0652 / +0.0545, all positive across the range."
)

translations[224] = (
    "EC asymmetric counter-evidence (N=500, source: tge_probe_ec.json): source-domain negative prototype lambda=0.5 is 0.4377; letting negative prototypes follow positive prototype extrapolation (EC) at same lambda=0.5 is 0.4197. EC only shows a higher peak at lambda=0.75 (0.4551),..."
)

translations[225] = (
    "Source-domain sample size (N=100, source: tge_probe_srcsize.json): 5 / 10 / 20 source-domain images per category -> 0.4456 / 0.4450 / 0.4461, 5 images already achieve 99% of full gain."
)

translations[226] = (
    "Efficiency (source: tge_cost_analysis.json): A-CSPC learnable parameters 0 (CoOp 4096, CLIP-Adapter 525312); GDINO-tiny detection 734 ms/image, after pre-filter average about 21 boxes..."
)

translations[227] = "v2 negative prototype pool size top-k ablation (COCO val2017 full image pool, beta=2.0, lambda=0.5, source: tge_v2_ablations_b2_l0.5.json):"

translations[228] = (
    "Top-3 is bit-by-bit consistent with the main table; k=1~10 range gain +0.058~+0.065, robust to negative prototype pool size (k=1 is already sufficient, k=10 slightly decreases)."
)

translations[229] = (
    "v2 score bucket recovery analysis (COCO val2017 full image pool, beta=2.0, lambda=0.5): divide proposals into 5 buckets by baseline gd score, count the average score of true targets in each bucket after being re-scored by A-CSPC. The true target score improvement in the lowest score bucket of six categories is all positive: a..."
)

translations[230] = (
    "v2 g2 gate decision (COCO 6 categories, beta=2.0, lambda=0.5): six categories g2 are all positive (+0.71~+0.82), so on the COCO forward domain gate is equivalent to full extrapolation; the selection value of gate is reflected in the 19-domain-pair validation of mixed directions (Section 3.4), here..."
)

translations[231] = "Supplementary Table S3. Information Usage Audit (Complete Version, Including Extended Columns Beyond Main Table 14)"

translations[232] = "Supplementary Table S4. Experimental Protocol (Reproduction Information;"

translations[233] = "Category mapping (source-domain prototype category -> COCO target category):"

translations[234] = (
    "COCO image sampling rule: from val2017 annotations, take all images per category that \"contain >=1 non-crowd annotation of that category\" (unified_eval.load_coco, in annotation file order), main table/ablation/significance all use full amount (airplane 97 / boat..."
)

translations[235] = (
    "Proposals and filtering: GDINO-tiny, post-process threshold=0 take all proposals -> pre-filter 0.03 (threshold sensitivity see Table 7) -> crop and discard small boxes with width or height <=10 px."
)

translations[236] = (
    "Models and weights: GDINO-tiny (HuggingFace IDEA-Research/grounding-dino-tiny, snapshot a2bb814dd30d776dcf7e30523b00659f4f141c71); CLIP..."
)

translations[237] = (
    "Hyperparameters and random seeds: beta=2.0, lambda=0.5 (value source see Section 5.5); text templates (s(x)=) \"an overhead image of {x}\", (d(x)=) \"a photo of {x}\"; negative prototypes are source-domain pool内 closest to positive prototypes by CLIP cosine..."
)

translations[238] = (
    "Evaluation: per-image IoU matching (greedy matching within each image, GT mutually exclusive per image) + global score ranking + dataset-level 11-point interpolation AP50 (unified_eval.compute_ap50_img); bootstrap resamples test images with replacement within categories (between categories相..."
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

output_path = r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx'
doc.save(output_path)
print(f"Saved: {output_path}")
print(f"Added {len(translations)} translations (remaining Chinese + supplementary)")

# Verify
doc2 = Document(output_path)
chinese_count = 0
for p in doc2.paragraphs:
    if p.text.strip():
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in p.text)
        if has_chinese:
            chinese_count += 1
print(f"Remaining Chinese paragraphs: {chinese_count}")
