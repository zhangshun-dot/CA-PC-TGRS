from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')

# ============================================================
# 1. Compress Section 4.5: Reverse Transfer (3 subsections -> 1 paragraph)
# ============================================================

# Change main 4.5 title (need to find it - it's before 4.5.1)
# Looking at structure, 4.5 title should be around paragraph 70
for i in range(68, 72):
    if '4.5' in doc.paragraphs[i].text and 'Reverse' in doc.paragraphs[i].text:
        p = doc.paragraphs[i]
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = '4.5 Reverse Transfer and Applicability Boundary'
        else:
            p.add_run('4.5 Reverse Transfer and Applicability Boundary')
        print(f'[{i}] Updated 4.5 title')
        break

# Clear subsection 4.5.1 title
p = doc.paragraphs[71]
for run in p.runs:
    run.text = ''
print('[71] Cleared 4.5.1 title')

# Compress 4.5.1 content into concise paragraph
new_45_content = (
    'In the reverse COCO-to-DIOR transfer (fine-to-coarse direction), the direction gate g2 automatically falls back to the B2 conservative mode, '
    'and source-domain visual prototypes directly transfer as optimal. '
    'The full extrapolation configuration yields -0.0244 AP50, while the B2 conservative mode (no extrapolation) achieves +0.0418 AP50, '
    'confirming that extrapolation is not beneficial in the fine-to-coarse direction. '
    'This is because fine-grained prototypes (car, bus, truck, motorcycle) already cluster near the coarse-grained centroid (vehicle) in the CLIP feature space, '
    'with cosine similarities exceeding 0.95, making direct source-prototype transfer sufficient. '
    'The direction gate g2 correctly identifies this boundary across 24 scene classes, with domain-direction discrimination accuracy of 24/24 and class-level decision accuracy of 18/24 (75%).'
)
p = doc.paragraphs[72]
for run in p.runs:
    run.text = ''
if p.runs:
    p.runs[0].text = new_45_content
else:
    p.add_run(new_45_content)
print(f'[72] Compressed 4.5 content ({len(new_45_content)} chars)')

# Clear subsection 4.5.2 title and content
p = doc.paragraphs[73]
for run in p.runs:
    run.text = ''
print('[73] Cleared 4.5.2 title')

p = doc.paragraphs[74]
for run in p.runs:
    run.text = ''
print('[74] Cleared 4.5.2 content')

# Clear subsection 4.5.3 title and content
p = doc.paragraphs[75]
for run in p.runs:
    run.text = ''
print('[75] Cleared 4.5.3 title')

p = doc.paragraphs[76]
for run in p.runs:
    run.text = ''
print('[76] Cleared 4.5.3 content')

# Clear Table 7a and its note (move to supplementary)
p = doc.paragraphs[77]
for run in p.runs:
    run.text = ''
print('[77] Cleared Table 7a caption')

p = doc.paragraphs[78]
for run in p.runs:
    run.text = ''
print('[78] Cleared Table 7a note')

# Clear remaining 4.5 paragraphs
p = doc.paragraphs[79]
for run in p.runs:
    run.text = ''
print('[79] Cleared 4.5 conclusion')

p = doc.paragraphs[80]
for run in p.runs:
    run.text = ''
print('[80] Cleared 4.5 data source')

# ============================================================
# 2. Compress Section 4.7: Component Attribution (move to supplementary)
# ============================================================

# Clear 4.7 title and content, replace with brief note
p = doc.paragraphs[87]
for run in p.runs:
    run.text = ''
if p.runs:
    p.runs[0].text = '4.7 Component Attribution Analysis (Supplementary)'
else:
    p.add_run('4.7 Component Attribution Analysis (Supplementary)')
print('[87] Updated 4.7 title')

p = doc.paragraphs[88]
for run in p.runs:
    run.text = ''
if p.runs:
    p.runs[0].text = 'Detailed component attribution analysis of reverse transfer, including the selectivity of text extrapolation direction and negative prototype construction method comparison, is provided in Supplementary Material.'
else:
    p.add_run('Detailed component attribution analysis of reverse transfer, including the selectivity of text extrapolation direction and negative prototype construction method comparison, is provided in Supplementary Material.')
print('[88] Compressed 4.7 content')

# Clear Table 10 (negative prototype comparison) - move to supplementary
p = doc.paragraphs[91]
for run in p.runs:
    run.text = ''
print('[91] Cleared Table 10 caption')

# ============================================================
# 3. Compress Section 4.8: Detector Replacement (move key finding to Limitations)
# ============================================================

p = doc.paragraphs[92]
for run in p.runs:
    run.text = ''
if p.runs:
    p.runs[0].text = '4.8 Detector Spectrum Experiment (Supplementary)'
else:
    p.add_run('4.8 Detector Spectrum Experiment (Supplementary)')
print('[92] Updated 4.8 title')

p = doc.paragraphs[93]
for run in p.runs:
    run.text = ''
if p.runs:
    p.runs[0].text = 'Detector spectrum experiments across Grounding DINO-tiny, GDINO-base, OWLv2, and YOLO-World are provided in Supplementary Material. The key finding is that A-CSPC achieves the largest gain on weakly-calibrated detectors (GDINO-tiny, baseline AP50=0.3707), while the gain approaches zero on strongly-calibrated detectors, consistent with the corrector positioning of the method.'
else:
    p.add_run('Detector spectrum experiments across Grounding DINO-tiny, GDINO-base, OWLv2, and YOLO-World are provided in Supplementary Material. The key finding is that A-CSPC achieves the largest gain on weakly-calibrated detectors (GDINO-tiny, baseline AP50=0.3707), while the gain approaches zero on strongly-calibrated detectors, consistent with the corrector positioning of the method.')
print('[93] Compressed 4.8 content')

# Clear Table 11 - move to supplementary
p = doc.paragraphs[96]
for run in p.runs:
    run.text = ''
print('[96] Cleared Table 11 caption')

# ============================================================
# 4. Adjust Discussion: Remove self-limiting language
# ============================================================

# 5.3 title: change from "Corrector Rather Than Amplifier" to more positive
p = doc.paragraphs[107]
for run in p.runs:
    run.text = ''
if p.runs:
    p.runs[0].text = '5.3 Detector Calibration and Correction Gain'
else:
    p.add_run('5.3 Detector Calibration and Correction Gain')
print('[107] Updated 5.3 title')

# 5.3 content: more positive framing
new_53 = (
    'A-CSPC improves proposal ranking quality for open-vocabulary detectors through prototype calibration. '
    'The method achieves the largest gain on detectors with lower baseline calibration (Grounding DINO-tiny, baseline AP50=0.3707, +0.0646 gain), '
    'where the proposal ranking quality has substantial room for improvement. '
    'On more strongly calibrated detectors, the baseline ranking is already near-optimal, resulting in smaller correction gains. '
    'This detector-dependent gain profile is consistent with the nature of re-ranking methods: the improvement is bounded by the quality of the fixed proposal pool. '
    'The method is particularly valuable for weakly-calibrated open-vocabulary detectors deployed in cross-domain settings, where target-domain training is unavailable.'
)
p = doc.paragraphs[108]
for run in p.runs:
    run.text = ''
if p.runs:
    p.runs[0].text = new_53
else:
    p.add_run(new_53)
print(f'[108] Updated 5.3 content ({len(new_53)} chars)')

# ============================================================
# 5. Update Limitations: integrate detector boundary and reverse transfer
# ============================================================

new_limitations = (
    'The method in this paper has several limitations. '
    'First, A-CSPC is a proposal re-ranking method that does not change the recall upper bound of the detector; the improvement is bounded by the coverage of the fixed proposal pool. '
    'Second, the method achieves the largest gain on weakly-calibrated open-vocabulary detectors; on strongly-calibrated detectors, the correction gain is smaller because the baseline ranking is already near-optimal. '
    'Third, the main configuration (beta=2.0, lambda=0.5) uses target-domain ground truth for hyperparameter tuning during development; the target-domain-agnostic configuration (beta=1.0, lambda=0.25) achieves +0.0166 AP50, and the physical-prior adaptive configuration achieves +0.0461 AP50 without target-domain annotations. '
    'Fourth, the extrapolation component is effective in the coarse-to-fine direction (source-domain coarse categories to target-domain fine categories); in the fine-to-coarse direction, the direction gate automatically falls back to conservative mode, and direct source-prototype transfer is optimal. '
    'Fifth, the physical prior prediction in Section 6 is exploratory evidence with rho=+0.699 across 37 classes and 6 domain pairs, but the cluster bootstrap 95% CI contains zero due to the limited number of independent domain pairs (K=6); further validation on more domain pairs is needed. '
    'Sixth, the main experimental target categories belong to classes seen during detector pretraining; the open-vocabulary nature is embodied in categories specified by text without target-domain training.'
)
p = doc.paragraphs[159]
for run in p.runs:
    run.text = ''
if p.runs:
    p.runs[0].text = new_limitations
else:
    p.add_run(new_limitations)
print(f'[159] Updated Limitations ({len(new_limitations)} chars)')

# ============================================================
# Save
# ============================================================
doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')
print()
print('Saved. Paper restructured.')
print()
print('=== Summary of Changes ===')
print('1. Section 4.5: 3 subsections -> 1 concise paragraph (reverse transfer + boundary)')
print('2. Section 4.7: Component attribution -> brief note, details to Supplementary')
print('3. Section 4.8: Detector spectrum -> brief note, key finding integrated, details to Supplementary')
print('4. Section 5.3: "Corrector Rather Than Amplifier" -> "Detector Calibration and Correction Gain" (positive framing)')
print('5. Section 7 Limitations: Integrated detector boundary, reverse transfer, hyperparameter source, physical prior scope')
print('6. Tables 7a, 10, 11: Cleared from main text, moved to Supplementary')
