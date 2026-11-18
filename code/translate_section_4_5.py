"""
Translate Section 4.5: Reverse Transfer and Granularity Mismatch Boundary
This is the core mechanism section with centroid clustering discovery
"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v3.docx')

translations = {}

# ============================================================
# SECTION 4.5: REVERSE TRANSFER (paras 70-80)
# ============================================================
translations[70] = "4.5 Reverse Transfer Validation and Granularity Mismatch Boundary: Centroid Clustering Structure and Direction Selectivity"

translations[71] = "4.5.1 Reverse Transfer Results"

translations[72] = (
    "Reverse COCO-to-DIOR transfer (3 DIOR target categories aggregated protocol, N=20 image protocol) is used to validate the "
    "applicability boundary of the A-CSPC extrapolation component. [Protocol note] Section 4.5 is cross-dataset overall reverse "
    "validation; Section 5.7 is a fixed COCO-to-NWPU, three-category, same CLIP encoding component control variable experiment. "
    "The two are not directly compared numerically. Results show a pattern opposite to the forward direction: conservative B2 "
    "(beta=0, pure source-domain visual prototype) averages +0.0418 with 3/3 categories positive, indicating that natural-scene "
    "source prototypes do not fail on the remote sensing target domain; the full extrapolation mode averages -0.0244 (3/3 "
    "categories negative), and pure text T4 drops even more (-0.0793). Under the COCO-to-DIOR six-row细分 protocol (Section 4.7), "
    "results show that negative prototypes have an amplifying effect under the wrong extrapolation direction; under the COCO-to-NWPU "
    "strict control variable protocol (Section 5.7), it is further shown that the text extrapolation direction is the main damage "
    "source, and negative prototypes further amplify this damage under the wrong direction."
)

translations[73] = "4.5.2 Centroid Clustering Structure"

translations[74] = (
    "Granularity mismatch and centroid clustering structure. The failure of reverse transfer is not merely \"direction selectivity,\" "
    "but has a deeper geometric root. In the CLIP joint feature space, the cosine similarities between fine-grained category prototypes "
    "(such as car, bus, truck, motorcycle) and their coarse-grained aggregated centroid (vehicle) are all higher than 0.95 (car 0.979, "
    "bus 0.958, truck 0.990, motorcycle 0.965), indicating that fine-grained prototypes are already highly clustered around the centroid "
    "(included angle approximately 8-17 degrees, average 13 degrees). This centroid clustering structure means that in \"fine-to-coarse\" "
    "transfer, source-domain fine-grained prototypes and target-domain coarse-grained prototypes are already highly aligned in feature space, "
    "and any form of extrapolation correction (whether text viewpoint direction or visual centroid contraction direction) introduces additional "
    "noise without producing gain - in the centroid contraction direction experiment, the displacement vector toward the centroid is extremely "
    "small because p_c approximately equals c_coarse, and its effect degenerates to approximately no movement, while \"no movement\" (B2 "
    "conservative) itself achieves the optimal gain of +0.0418."
)

translations[75] = "4.5.3 Granularity Mismatch Boundary and Three-Direction Ablation"

translations[76] = (
    "Granularity interpretation of direction selectivity. The above findings elevate \"direction selectivity\" to the more essential "
    "\"granularity mismatch boundary\": A-CSPC's text direction extrapolation produces value when correcting \"source-domain coarse-grained "
    "to target-domain fine-grained\" (centroid to circumference, with bridgeable semantic gaps); while in \"source-domain fine-grained to "
    "target-domain coarse-grained\" (circumference to centroid, fine-grained prototypes already clustered around the centroid), the premise "
    "of correction (existence of bridgeable gaps) does not hold, and extrapolation反而 introduces noise. This boundary is determined by the "
    "geometric structure of CLIP space, independent of the choice of specific extrapolation direction - multi-template direction calibration "
    "(reducing 13.7% damage), target-domain negative prototype replacement (reducing 27.3% damage), and visual centroid contraction (damage "
    "reduced to approximately 0) all cannot exceed B2 conservative +0.0418, and the three jointly validate the geometric conclusion that "
    "\"reverse transfer requires no correction.\""
)

translations[77] = "Table 7a. Reverse transfer three-direction ablation results (COCO-to-DIOR, 3-category aggregated, alpha=2.5)"

translations[78] = (
    "Note: The gain values for rows B/C/D are estimated based on the damage reduction ratios reported in Section 4.5, with precise values "
    "subject to supplementary material Table S-X. The three-direction ablation jointly validates that the optimal solution for reverse transfer "
    "is no correction (B2 conservative +0.0418), and no form of extrapolation correction can exceed conservative mapping."
)

translations[79] = (
    "Therefore, A-CSPC should be positioned as: a cross-domain prototype corrector for a specific granularity direction (coarse-to-fine); "
    "in the fine-to-coarse direction, the method automatically falls back to B2 conservative mode (g2 <= 0), and source-domain visual "
    "prototypes directly transfer as optimal. The domain direction discrimination of the direction gate g2 achieves 19/19 accuracy on 19 "
    "domain pairs, and class-level decision accuracy is 16/19 (the former judges domain direction, the latter judges whether specific categories "
    "should be extrapolated). (Reverse transfer protocol quick reference: -0.0244 is COCO-to-DIOR 3-category aggregated N=20 images / -0.0866 "
    "is six-row细分 protocol / -0.239 is COCO-to-NWPU three-category control variable experiment, the three have different protocols and cannot be "
    "directly compared)."
)

translations[80] = (
    "Data source for this subsection is tge_probe_rev.json (N=20 images/category, DIOR target 3 categories aggregated, image-safe per-image "
    "AP50, COCO source side 500 images); the three means are consistent with the average of per-category gains (extrapolation term back-calculated "
    "from JSON 4 decimal places as -0.0243, main text -0.0244 differs by one last digit, from unrounded precision). This protocol differs from "
    "Section 4.3 Table 4 (N=100, 6-row split), and image pool, category aggregation granularity, and baseline cannot be directly compared: the "
    "vehicle aggregated row dilutes the collapse of the four vehicle subcategories to -0.014, while Table 4 split exposes truck/bus/motorcycle "
    "at -0.14 to -0.17."
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

output_path = r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v4.docx'
doc.save(output_path)
print(f"Saved: {output_path}")
print(f"Added {len(translations)} translations (Section 4.5)")
print(f"Total translated: {67 + len(translations)} paragraphs")
