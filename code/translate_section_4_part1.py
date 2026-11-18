"""
Translate Section 4 Part 1: 4.1-4.4 (Datasets, Main Results, Failure Diagnosis, Ablation)
"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2.docx')

translations = {}

# ============================================================
# SECTION 4: EXPERIMENTS (paras 41-69)
# ============================================================
translations[41] = "4. Experiments"

translations[42] = (
    "The experimental section is organized around the three pillars of diagnosis-repair-boundary: "
    "Section 4.3 layered failure diagnosis corresponds to the P1 diagnosis pillar, identifying the true "
    "dimension of prototype failure; Section 4.4 asymmetric construction ablation corresponds to the P2 "
    "repair pillar, verifying the necessity of the positive/negative prototype asymmetric construction; "
    "Section 4.5 reverse transfer verification, Section 4.7 reverse component attribution, and Section 4.8 "
    "detector spectrum correspond to the P3 boundary pillar, systematically delineating the repairable "
    "boundary and applicability scope of the method."
)

translations[43] = "4.1 Datasets and Implementation Protocol"

translations[44] = (
    "Source domain datasets: DIOR [4] (20 remote sensing object categories, used for main experiments), "
    "NWPU [23], DOTA [24], VisDrone [25] (used for cross-source replication). Target domain datasets: "
    "COCO [5] (main target domain, 80 natural scene categories), VOC [26], NWPU (reverse transfer). "
    "The main experiment selects 3 mapped source categories on the DIOR side (airplane, ship, vehicle), "
    "mapped to 6 target categories on the COCO side (airplane, boat, car, truck, bus, motorcycle); "
    "where vehicle is a coarse-grained source category mapped to four fine-grained COCO categories "
    "(car/truck/bus/motorcycle), and ship maps to boat. The 6-category setting is a hard constraint of "
    "shared categories between remote sensing and natural scenes (DIOR has only 20 categories, and the "
    "mappable shared categories in COCO are limited), rather than a lazy choice; the VOC-to-COCO 20-category "
    "extension experiment (Section 4.2.1) validates the generality of the method on a larger category scale."
)

translations[45] = (
    "Detectors: Grounding DINO-tiny [2] (main detector, baseline AP50=0.3707), GDINO-base, OWLv2 [3], "
    "YOLO-World [12] (detector spectrum experiments). Visual encoder: CLIP ViT-B/32 [1], with proposal "
    "features extracted after ROI Align. Evaluation metrics: dataset-level 11-point AP@0.5 (main metric), "
    "with AP@[0.5:0.95] reported as reference. The main configuration (beta=2.0, lambda=0.5) uses "
    "target-domain COCO GT for hyperparameter tuning (disclosed in Section 4.9); the target-domain-agnostic "
    "configuration (1.0, 0.25) is selected on the DIOR-to-VOC development domain."
)

translations[46] = (
    "Efficiency and parameter analysis: A-CSPC is a pure inference method with zero learnable parameters, "
    "requiring no training or fine-tuning at inference. Inference time breakdown (RTX 5060, CUDA): "
    "GDINO-tiny detection 734.2 ms/image, proposal CLIP feature encoding 16.9 ms/proposal (approximately "
    "21 proposals/image after pre-filtering, about 355 ms/image), A-CSPC scoring 0.0001 ms/proposal "
    "(negligible), end-to-end approximately 1090 ms/image (about 1.5x detection time). Comparison method "
    "parameters: CoOp prompt learning 4096 parameters (requires 10 epochs training), CLIP-Adapter residual "
    "MLP + lambda total 525,312 parameters (requires 20 epochs training), A-CSPC 0 parameters (zero training). "
    "The zero-training characteristic of A-CSPC is particularly suitable for scenarios with limited computing "
    "power or inaccessible target-domain training data."
)

translations[47] = "4.2 Main Results and Setting Groups"

translations[48] = (
    "Main results (DIOR-to-COCO, 6 categories, GDINO-tiny): the target-domain-tuned configuration A-CSPC(2.0,0.5) "
    "achieves +0.0646 AP50 gain, with 6/6 categories positive (airplane +0.0117, boat +0.0253, car +0.0809, "
    "truck +0.0774, bus +0.0735, motorcycle +0.1189). The target-domain-agnostic configuration A-CSPC(1.0,0.25) "
    "yields +0.0166 gain, but boat turns negative (this configuration is selected from the high-baseline development "
    "domain VOC, whose beta response surface is too flat and pathological, with failed parameter selection, see "
    "Section 4.8 and supplementary material), indicating that target-domain-agnostic parameter selection remains an "
    "open problem. Baseline comparison: pure text prototype -0.085, pure visual prototype -0.0005 (both negative, "
    "validating the necessity of visual anchoring), TGE component control (N=100 protocol, same as Section 4.4 "
    "ablation) +0.0239, source-domain visual prototype baseline (beta=0, full 6-category protocol) -0.071, the "
    "point estimate of A-CSPC's asymmetric construction is higher than symmetric extrapolation methods. Setting groups..."
)

translations[49] = (
    "Statistical significance testing: 1000 paired image-level bootstrap (seed=42, 90 test images, COCO official "
    "per-image AP50 protocol) on the main results, 95% confidence interval of 6-category mean gain is [0.052, 0.082], "
    "p=0.001 (one-sided test, only 1 out of 1000 resamples has gain <= 0). Per-category significance: airplane p=0.066 "
    "(marginal significance, 935/1000 positive), boat p=0.025, car/truck/bus/motorcycle all p=0.001 (1000/1000 "
    "positive). The NWPU source domain replication experiment also undergoes bootstrap validation, with mean gain 95% "
    "CI=[0.045, 0.074], p=0.001, indicating that the gain is statistically significant across different source domains."
)

translations[50] = (
    "Comparison with existing domain adaptive re-ranking methods (same proposal pool, same CLIP features, fair protocol). "
    "To avoid unfair comparison due to inconsistent information budgets, results are divided into two groups by \"whether "
    "target-domain annotations are used\": Table 2a is the main comparison under the target-domain-annotation-free setting "
    "(all methods do not use target-domain GT), Table 2b is the upper-bound reference using target-domain GT for tuning, "
    "not participating in the main comparison."
)

translations[51] = (
    "Table 2a. Main comparison under target-domain-annotation-free setting (all methods do not use target-domain GT; "
    "DANN uses unlabeled proposal features from 200 target-domain images for training, reporting mean +/- standard "
    "deviation of 5 random seeds (1/7/42/2026/2027); CORAL uses unlabeled proposal features to compute covariance; "
    "T4 is a degraded configuration of the method family in this paper (re-ranking using only target-domain text embeddings), "
    "listed for component contribution decomposition, not an external comparison method; AP50, 11-point interpolation)."
)

translations[52] = (
    "DANN training instability analysis: to exclude the质疑 that \"domain adversarial failure is due to insufficient "
    "training data or single-run deviation\", two additional experiments were conducted: (1) target-domain training image "
    "count scanned from 10 to 200 (10/50/100/200 images), 6-category mean gains are -0.0958, -0.0087, -0.0049, +0.0009 "
    "(single run); (2) 5 random seeds for the 200-image configuration, 6-category mean gain is -0.0551 +/- 0.1090. Results "
    "show that DANN training is extremely unstable under this setting: some seeds learn near-identity transformations (AP "
    "identical to baseline), some seeds completely collapse (e.g., airplane category seed=42 gain -0.4333). The multi-seed "
    "mean is negative (-0.0551), but the standard deviation reaches 0.1090 (2.0 times the absolute value of the mean), 95% "
    "confidence interval [-0.1904, +0.0802] crosses zero, so it cannot be判定 as significantly harming performance; the robust "
    "conclusion is that domain adversarial training under this setting not only has no gain, but also does not exceed CORAL "
    "(+0.0210) and A-CSPC physical prior adaptive under any of the 5 seeds..."
)

translations[53] = (
    "Net increment analysis: the difference between A-CSPC main configuration (+0.0646) and pure text prototype T4 (+0.0357) "
    "is +0.0289, i.e., the net increment of the asymmetric prototype construction accounts for 44.7% of the total gain, with the "
    "remaining 55.3% coming from the simple operation of \"re-ranking using target-domain text embeddings\". This decomposition "
    "shows the contribution of each component of the method, avoiding overclaiming the role of asymmetric construction. Under the "
    "target-domain-annotation-free setting, the net increment of A-CSPC physical prior adaptive (automated +0.0461) compared to "
    "T4 (+0.0357) is +0.0104, indicating that physical-prior-driven adaptive parameter control still brings stable gain under the "
    "annotation-free setting, and completely requires no manual annotation or training parameters."
)

translations[54] = (
    "Table 2b. Upper-bound reference using target-domain GT for tuning (A-CSPC main configuration uses COCO GT for beta/lambda "
    "grid search, only as performance upper-bound reference, not participating in Table 2a main comparison; AP50, 11-point "
    "interpolation)."
)

translations[55] = "4.2.1 Extended Category Validation: VOC-to-COCO 20 Categories"

translations[56] = (
    "To validate that A-CSPC does not depend on the special structure of DIOR-to-COCO 6 categories, an extended category "
    "validation was conducted on VOC-to-COCO 20 categories. VOC2007 serves as the source domain, COCO val2017 as the target "
    "domain, covering 20 common object categories (including animals, vehicles, furniture, electronics, etc.), with category "
    "diversity much higher than DIOR-to-COCO 6 categories. Using the same proposal pool and CLIP features, the fixed-parameter "
    "configuration (beta=1.0, lambda=0.25) is selected on the DIOR-to-VOC development domain, and the physical prior adaptive "
    "configuration (alpha=1.5) uses automatically assigned class-level priors."
)

translations[57] = (
    "Table VOC20. VOC-to-COCO 20-category extended validation (VOC2007 source domain, COCO val2017 target domain, GDINO-tiny "
    "proposal pool; fixed parameters beta=1.0, lambda=0.25 selected on DIOR-to-VOC development domain; adaptive prior alpha=1.5 "
    "uses automatically assigned class-level priors; AP50, 11-point interpolation)."
)

translations[58] = (
    "Results show that A-CSPC achieves stable gain on VOC-to-COCO 20 categories: fixed-parameter configuration 20-category average "
    "+0.0301 (19/20 categories positive), physical prior adaptive configuration 20-category average +0.0328 (18/20 categories "
    "positive). Adaptive prior has further improvement compared to fixed parameters (+0.0328 vs. +0.0301), validating the "
    "effectiveness of physical-prior-driven per-class parameter control on broader categories. The two negative-gain categories "
    "(potted plant -0.0052, train -0.0048) both have magnitude less than 0.006,属于 acceptable boundary cases; among them, potted "
    "plant is -0.0298 under fixed parameters, and adaptive prior improves it to -0.0052, indicating that physical prior risk "
    "modulation has a repairing effect on difficult categories. This extended category validation indicates that the gain of A-CSPC "
    "does not depend on the special structure of DIOR-to-COCO 6 categories, and remains stable and effective on broader 20 categories "
    "and more diverse category types."
)

translations[59] = "4.3 Failure Diagnosis and Proposal Pool Upper Bound"

translations[60] = (
    "Table 4. Layered failure diagnosis results (ranking AUC and proposal pool recall reported in layers; n is the number of "
    "categories per layer, B2 is beta=0 source prototype, T4 is pure text prototype, oracle beta is per-class optimal extrapolation "
    "strength)."
)

translations[61] = (
    "Layered failure diagnosis results: the maximum achievable recall of the proposal pool R_max (proportion of category GT boxes "
    "covered by proposals with IoU >= 0.5, N=100 pre-filtering protocol) averages 0.82 across 6 categories (range 0.71-0.91), "
    "indicating that the proposal pool contains most true targets, with re-ranking gain space. Source prototype ranking AUC on bus "
    "and motorcycle is 0.20 and 0.29 respectively (below random 0.5), on airplane and ship is 0.62 and 0.58. Key finding: the "
    "correlation coefficient between the ranking improvement from extrapolation Delta AUC and final AP gain is rho=+0.577 (32 "
    "categories, 5 target domains), while the original source prototype AUC has no significant correlation with gain (rho=+0.12), "
    "validating the diagnostic conclusion that \"extrapolation improvement rather than failure degree is predictive\"."
)

translations[62] = (
    "Two counter-evidences for the asymmetric principle. First, EC ablation (letting negative prototypes follow positive prototype "
    "extrapolation) is lower than source-domain negative prototypes under the same lambda (N=500: 0.4197 vs. 0.4377); EC only "
    "shows higher peaks when selecting parameters across lambda, not constituting validity evidence. Second, NegPrompt-style trained "
    "negative prompts (source-domain training, N=100 protocol) is 0.4345 (+0.0536), lower than zero-training A-CSPC - negative "
    "prototypes are not a substitute for learnable negative prompts, and giving training is even worse; this row is 0.4175 under the "
    "full unified protocol (Table 1a). The two counter-evidences jointly support the asymmetric construction of \"positive prototype "
    "extrapolation, negative prototype retained in source domain space\"."
)

translations[63] = "Table 5. Negative prototype pool size top-k ablation experiment"

translations[64] = "4.4 Ablation Experiments"

translations[65] = (
    "Ablation experiments verify the necessity of each component of the asymmetric construction (N=100 protocol, 6 categories, "
    "GDINO-tiny, beta=2.0, lambda=0.5 main configuration): (1) removing positive prototype extrapolation (Eq. 1) (beta=0, only "
    "source-domain visual prototype + negative prototype): gain drops from +0.0652 to -0.0005, indicating that text direction "
    "extrapolation is the core gain source, and without extrapolation the source-domain visual prototype can hardly improve ranking; "
    "(2) removing negative prototype (lambda=0, only positive prototype extrapolation (Eq. 1)): pure extrapolation gain +0.0239 under "
    "N=100 protocol (37% of total gain), negative prototype conditional increment +0.0413 (63%), total gain +0.0652 can be completely "
    "decomposed into the sum of both, indicating that source-domain-space negative prototypes play an important role in background "
    "suppression; (3) symmetric extrapolation (negative prototypes follow positive prototype extrapolation (Eq. 1)): mean AP drops "
    "from 0.4377 to 0.4197 under the same lambda (N=500; 21-category/3-domain protocol 0.4455 to 0.4192), validating the hypothesis "
    "that negative prototype extrapolation introduces noise, and asymmetric construction (positive prototype extrapolation (Eq. 1), "
    "negative prototype retained in source domain) is the key design. Minimal baseline comparison (N=100 protocol..."
)

translations[66] = "Table 6. Extrapolation strength beta scan experiment (N=100 protocol)"

translations[68] = "Table 7. Negative prototype weight lambda scan experiment (N=500 per-image AP50)"

translations[69] = (
    "Additional ablation experiments: (a) pre-filtering threshold sensitivity: tested at thresholds 0.01/0.03/0.05 respectively, "
    "6-category mean gains are +0.0615/+0.0567/+0.0517, all maintaining significant positive gain, indicating that the method is "
    "insensitive to the pre-filtering threshold and robust over a wide range. When the threshold decreases, proposals increase and "
    "baseline AP decreases, but the absolute gain of A-CSPC反而 increases, indicating that the method has larger correction space on "
    "low-quality proposal pools. (b) source-domain sample size sensitivity: using 5/10/20 annotated boxes per category from DIOR "
    "source domain to construct prototypes, 6-category mean gains are +0.0647/+0.0641/+0.0652, with differences among the three less "
    "than 0.001, indicating that the method has extremely low demand for source-domain annotation data, requiring only 5 samples per "
    "category to achieve gain comparable to 20 samples. This characteristic makes the method particularly suitable for cross-domain "
    "scenarios with scarce source-domain annotations."
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

output_path = r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v3.docx'
doc.save(output_path)
print(f"Saved: {output_path}")
print(f"Added {len(translations)} translations (Section 4.1-4.4)")
print(f"Total translated: {39 + len(translations)} paragraphs")
