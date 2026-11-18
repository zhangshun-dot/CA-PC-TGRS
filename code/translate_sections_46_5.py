"""
Translate Sections 4.6-4.9 and Chapter 5 (Discussion)
"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v4.docx')

translations = {}

# ============================================================
# SECTION 4.6: SOURCE PROTOTYPE LIBRARY SENSITIVITY (81-86)
# ============================================================
translations[81] = "4.6 Source Prototype Library Sensitivity: NWPU, DOTA, and VisDrone as Source Domains"

translations[82] = (
    "Cross-source replication experiments validate the robustness of the method to the source prototype library: "
    "NWPU, DOTA, and VisDrone are used as source domains respectively, transferred to the COCO target domain. "
    "Results: NWPU VHR-10 to COCO +0.0579 (6/6 mapped categories positive gain; per-category: airplane +0.0099 / "
    "harbor +0.0297 / ship +0.0734 / vehicle-car +0.0563 / vehicle-truck +0.0669 / vehicle-bus +0.1111, all >= +0.009), "
    "DOTA to COCO +0.0585 (6/6..."
)

translations[84] = (
    "The above results indicate that A-CSPC is robust to the source prototype library (gain remains positive and gain "
    "structure is similar after changing source domain), but all four source domains are transferred to the same COCO target "
    "domain, and beta/lambda are determined by COCO tuning - this subsection delineates \"source prototype library sensitivity,\" "
    "not independent target domain generalization; the conclusion should be limited to \"replication from multiple remote sensing/UAV "
    "source domains to the same natural scene target domain.\""
)

translations[85] = "Table 8. Cross-source domain category mapping table (DIOR/NWPU/DOTA/VisDrone to COCO)"

translations[86] = "Table 9. Three-source-domain cross-source replication results (DIOR/NWPU/DOTA to COCO, per-category gain and mean; VisDrone is 4 categories with different granularity, listed separately in main text)"

# ============================================================
# SECTION 4.7: REVERSE TRANSFER COMPONENT ATTRIBUTION (87-91)
# ============================================================
translations[87] = "4.7 Component Attribution of Reverse Transfer: Selectivity of Text Extrapolation Direction"

translations[88] = (
    "Component attribution of reverse transfer (COCO-to-NWPU, 3-category control variable experiment): Configuration A "
    "(pure source-domain visual prototype, no extrapolation) average gain +0.100, indicating that source-domain visual prototype "
    "re-ranking is still effective in reverse tasks; Configuration B (A + negative prototype) average gain +0.0767, negative "
    "prototypes have slight side effects; Configuration C (A + positive prototype text extrapolation) average gain -0.081, "
    "performance明显 drops after adding text extrapolation; Configuration D (full A-CSPC) average gain -0.239, negative prototypes "
    "further amplify damage under the wrong extrapolation direction. Beta scan (beta in {0, 0.25, 0.5, 1.0, 1.5, 2.0}, lambda=0) "
    "shows that performance持续 drops after beta > 0, supporting the conclusion that \"photo-to-overhead..."
)

translations[90] = "Figure 3. Beta collapse point aligned with source-domain autonomous round-trip residual inflection point (beta=2 safe, beta=4 explosion; left: target domain AUC vs. beta, right: source-domain autonomous round-trip residual vs. beta, both inflection points between beta=2-4)"

translations[91] = "Table 10. Negative prototype construction method comparison (top-k cosine nearest / random / all)"

# ============================================================
# SECTION 4.8: DETECTOR REPLACEMENT EXPERIMENT (92-97)
# ============================================================
translations[92] = "4.8 Detector Replacement Experiment: Corrector Boundary of A-CSPC"

translations[93] = (
    "Detector spectrum experiments (N=100/category subset protocol, consistent with main table; proposal GT coverage rate "
    "statistics based on original output of each detector, different detectors have different proposal counts and confidence "
    "thresholds, coverage rate ranges differ) validate the corrector boundary of A-CSPC: GDINO-tiny (baseline 0.3707) gain "
    "+0.0646, GDINO-base (baseline 0.4037) gain +0.0790, YOLO-World (baseline 0.4120) gain +0.0063, OWLv2 (baseline 0.6800) "
    "gain -0.0316. This result is not simply \"weak detectors have large gain,\" but is continuously correlated with the headroom "
    "(1-AP) of the detector baseline..."
)

translations[95] = "Figure 4. A-CSPC gain on weak-to-strong detector spectrum (GDINO-tiny +0.0646, GDINO-base +0.0790, YOLO-World +0.0063, OWLv2 -0.0316, showing non-monotonic corrector boundary)"

translations[96] = "Table 11. Detector spectrum experiment (corrector boundary: weak-to-strong detectors)"

translations[97] = "Table 12. Multi-target-domain result comparison (COCO/VOC/NWPU, 21 categories; grouped by target domain information budget)"

# ============================================================
# SECTION 4.9: HYPERPARAMETER SOURCE (98-100)
# ============================================================
translations[98] = "4.9 Hyperparameter Source and Information Usage Protocol"

translations[99] = (
    "Configuration selection audit clarifies hyperparameter sources: the main configuration (beta=2.0, lambda=0.5) is determined "
    "through 6-category grid search on target-domain COCO, belonging to target-domain tuning configuration, not claiming zero-shot. "
    "Beta scan shows maximum gain at beta=2.0 (+0.0318 at beta=1.0, -0.0684 at beta=4.0 entering collapse zone), lambda scan "
    "(N=500 per-image AP50) shows mean AP=0.3910 at lambda=0, corresponding to N=500 per-category baseline 0.3728, gain +0.0182; "
    "lambda=0.5 is the 6-category common peak 0.4377 (+0.0649); lambda=0.75 shows cliff-like drop to 0.3395; lambda=1.0 degenerates "
    "to baseline 0.3728 bit by bit..."
)

translations[100] = "Table 13 (Section 4.5 COCO-to-DIOR protocol, N=20 images/category, 3-category aggregated) Reverse transfer six-row细分 protocol results (COCO-to-DIOR)"

# ============================================================
# CHAPTER 5: DISCUSSION (101-119)
# ============================================================
translations[101] = "5. Discussion"

translations[102] = "5.1 Why Negative Prototypes Need to Remain in Source Domain Space"

translations[103] = (
    "Negative prototypes are retained in source domain space rather than extrapolated to the target domain, based on two observations: "
    "(1) negative prototypes represent inter-class confusion category features (top-k categories closest in cosine to positive prototypes "
    "in the source domain pool), and the statistical characteristics of inter-class confusion structures differ less between source and "
    "target domains than foreground categories, with source-domain-space negative prototypes already able to effectively suppress inter-class "
    "confusion; (2) extrapolating negative prototypes introduces target-domain background noise,反而 reducing discriminability for true "
    "targets. Ablation experiments validate: symmetric extrapolation (negative prototypes follow extrapolation) reduces mean AP under the "
    "same lambda (0.4377 to 0.4197, N=500), and removing negative prototypes shrinks total gain by about 60% (+0.0652 to +0.0239, N=100)."
)

translations[105] = "5.2 Cross-Source Prototype Library Replication and Method Boundary"

translations[106] = (
    "Three-source-domain cross-source replication experiments (DIOR/NWPU/DOTA (VisDrone is 4 categories with different granularity, "
    "listed separately in main text) to COCO) all achieve positive gain (+0.0534 to +0.0646), validating the robustness of A-CSPC to "
    "the source prototype library. Method boundary: gain magnitude varies with source-domain category granularity and image quality, with "
    "coarse-grained, high-quality source domains (DIOR) having the highest gain, and fine-grained, low-quality source domains (VisDrone) "
    "having the lowest gain."
)

translations[107] = "5.3 Detector Dependence: Corrector Rather Than Amplifier"

translations[108] = (
    "A-CSPC is a corrector rather than an amplifier: its gain comes from improving proposal ranking of weakly-calibrated detectors, "
    "rather than enhancing the feature extraction capability of the detector itself. Detector spectrum experiments show a non-monotonic "
    "boundary: weak detectors (GDINO-tiny/base) have large gain, medium detectors (YOLO-World) have small gain, strong detectors (OWLv2) "
    "have negative gain. This result clarifies the applicable scope of the method. Specifically, the gain of A-CSPC comes from the "
    "simultaneous satisfaction of two conditions: (i) detector proposal ranking has systematic bias (weak calibration), (ii) source-domain "
    "visual prototypes can provide effective ranking correction signals. When the detector itself is already well-calibrated (such as OWLv2, "
    "baseline AP50=0.6..."
)

translations[109] = "5.4 Formal Analysis: Corrector Boundary and Asymmetric Construction"

translations[110] = (
    "Formal analysis gives three propositions, characterizing the theoretical basis of corrector boundary and asymmetric construction. "
    "Observation 1 (recall upper bound): when the proposal pool is fixed, the maximum achievable recall of any re-ranking method remains "
    "unchanged, and the AP gain upper bound is determined by the maximum recall of the proposal pool. Intuition: re-ranking can only improve "
    "ranking, cannot recover missed targets. Observation 2 (saturation boundary): when baseline ranking is approximately saturated, the gain "
    "upper bound of any re-ranking method approaches zero. Intuition: when ranking is already optimal, any change introduces noise. Observation "
    "3 (asymmetric rationality): the asymmetric construction of positive prototype extrapolation and negative prototype non-extrapolation is "
    "optimal when source-domain background statistics and target-domain background statistics differ greatly. Intuition: when background categories "
    "have large cross-domain differences, positive prototypes need to be extrapolated to adapt to target-domain foreground distribution..."
)

translations[111] = "5.5 Action Level of g2: Domain Direction Gate Rather Than Class-Level Selector"

translations[112] = (
    "g2 is a domain direction gate rather than a class-level selector: its role is to judge whether the text extrapolation direction of the "
    "entire domain pair is effective, rather than deciding whether to extrapolate for each category individually. When g2 > 0, extrapolation "
    "is performed along the source-to-target direction; when g2 <= 0, it falls back to pure source-domain prototypes. In reverse transfer "
    "experiments, g2 <= 0 (photo-to-overhead direction is invalid), and the method falls back to pure source-domain visual prototypes, avoiding "
    "damage from wrong extrapolation. The domain direction discrimination of g2 achieves 19/19 accuracy on 19 domain pairs, and class-level "
    "decision accuracy is 16/19 (the former judges domain direction, the latter judges whether specific categories should be extrapolated)."
)

translations[113] = "5.6 Reproducibility and Information Usage Audit"

translations[114] = (
    "Information usage audit clarifies the information budget of each configuration: (i) the main A-CSPC configuration does not use target-domain "
    "images or annotations at inference, but hyperparameters beta/lambda are tuned by target-domain GT (Section 4.9); (ii) the target-domain-agnostic "
    "configuration (1.0, 0.25) is selected on the development domain DIOR-to-VOC, and does not contact the target domain at test time; (iii) realized "
    "gamma per-class control uses target-domain baseline AP (requiring GT), belonging to oracle upper-bound analysis. All hyperparameter sources, random "
    "seeds, and proposal cache processing methods are disclosed in supplementary material, ensuring reproducibility."
)

translations[115] = "Table 14. Information usage audit table (target domain information budget of each configuration)"

translations[116] = "5.7 Component Attribution in Reverse Transfer: Visual Re-Ranking Has Direction Robustness, While Text Extrapolation Has Direction Selectivity"

translations[118] = "Figure 5. Selection of direction gate g2 among three strategies (19 categories, 13 extrapolated, decision accuracy 16/19; left: mean comparison of full extrapolation / full B2 / gate three strategies, right: 19-category extrapolation - B2 gain, blue = gate correct, red = misjudgment)"

translations[119] = (
    "Component attribution experiments of reverse transfer (COCO-to-NWPU) (Section 4.7) show that: visual prototype re-ranking has direction "
    "robustness (pure source-domain visual prototypes still achieve +0.100 average gain in reverse tasks), while text extrapolation has direction "
    "selectivity (gain drops to -0.081 after adding photo-to-overhead text extrapolation, and full A-CSPC further drops to -0.239). Beta scan shows "
    "that performance持续 drops after beta > 0, supporting the conclusion that the direction itself is not applicable. Therefore, A-CSPC should be "
    "positioned as: an asymmetric prototype corrector for specific domain directions (overhead-to-photo) and coarse-to-fine granularity mismatch, "
    "rather than a general extrapolation method symmetric to transfer direction. Reverse negative results serve as method boundary..."
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

output_path = r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v5.docx'
doc.save(output_path)
print(f"Saved: {output_path}")
print(f"Added {len(translations)} translations (Sections 4.6-4.9 + Chapter 5)")
print(f"Total translated: {78 + len(translations)} paragraphs")
