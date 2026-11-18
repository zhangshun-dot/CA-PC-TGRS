from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')

# Fix reference Chinese annotations
doc.paragraphs[169].runs[0].text = '[7] Zhang X, Wei F, Wang Y, et al. UPRE: Zero-shot domain adaptation for object detection via unified prompt and representation enhancement[C]//IEEE/CVF International Conference on Computer Vision (ICCV). 2025. arXiv:2507.00721. (Different setting: zero-shot domain adaptation detection method requiring training, only...'

doc.paragraphs[175].runs[0].text = '[13] Zhang Y, Feng Y, Han R, et al. ExDet: Open-domain open-vocabulary detection with cross-modal extrapolation and rectification[J/OL]. arXiv:2606.09360, 2026. (TGE is the Text-Guided Extrapolation component; this paper only migrates its TGE component, not including DCR and ExRPN requiring training)[C]'

doc.paragraphs[188].runs[0].text = '[26] Cheng G, Han J, Zhou P, et al. Multi-class geospatial object detection and geographic image classification based on collection of part detectors[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2014, 98: 119-132. (NWPU VHR-10 dataset)'

doc.paragraphs[196].runs[0].text = '[34] Zhu P, Wen L, Du D, et al. Detection and tracking meet drones challenge[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, 44(11): 7380-7399. (VisDrone dataset)'

doc.paragraphs[197].runs[0].text = '[35] Li T, Pang G, Bai X, et al. Learning transferable negative prompts for out-of-distribution detection[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024. arXiv:2404.03248. (NegPrompt; this paper reproduces its negative prompt learning idea in the detection re-scoring setting, K=16, M=3)'

# Fix supplementary material incomplete translations
doc.paragraphs[237].runs[0].text = 'Hyperparameters and random seeds: beta=2.0, lambda=0.5 (value source see Section 5.5); text templates (s(x)=) "an overhead image of {x}", (d(x)=) "a photo of {x}"; negative prototypes are the top-k closest to positive prototypes by CLIP cosine in the source domain pool...'

doc.paragraphs[238].runs[0].text = 'Evaluation: per-image IoU matching (greedy matching within each image, GT mutually exclusive per image) + global score ranking + dataset-level 11-point interpolation AP50 (unified_eval.compute_ap50_img); bootstrap resamples test images with replacement within categories (between categories independent...'

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')

# Verify
doc2 = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')
chinese_count = 0
for p in doc2.paragraphs:
    if p.text.strip():
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in p.text)
        if has_chinese:
            chinese_count += 1
            print(f'Still Chinese: {p.text[:80]}')
print(f'Remaining Chinese paragraphs: {chinese_count}')
print('Translation complete!')
