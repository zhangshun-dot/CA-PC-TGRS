from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')

# Fix remaining Chinese cells
fixes = [
    # (table_idx, row_idx, col_idx, new_text)
    (3, 0, 5, 'T4 Text-only'),
    (8, 3, 1, 'plane→airplane; ship→boat; vehicle (large-vehicle + small-vehicle merged)→car'),
    (12, 2, 1, 'None (zero-shot)'),
    (12, 3, 1, 'Per-image (entropy minimization)'),
    (12, 4, 1, 'Streaming online (confidence cache)'),
    (12, 5, 1, 'All unlabeled test set'),
    (12, 6, 1, 'Per-box (augmented view marginalization)'),
    (13, 0, 1, '6-row Mean'),
    (14, 4, 1, 'Inference'),
]

for t_idx, r_idx, c_idx, new_text in fixes:
    table = doc.tables[t_idx]
    cell = table.rows[r_idx].cells[c_idx]
    # Clear and set new text
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.text = ''
    if cell.paragraphs and cell.paragraphs[0].runs:
        cell.paragraphs[0].runs[0].text = new_text
    else:
        cell.text = new_text
    print(f'Fixed Table {t_idx}, Row {r_idx}, Col {c_idx}: {new_text[:50]}')

# Fix Table 14 Row 8 Col 5 - check and fix
cell = doc.tables[14].rows[8].cells[5]
original = cell.text
if re.search(r'[\u4e00-\u9fff]', original):
    # Translate remaining Chinese
    new_text = original.replace('测试域先验值查看', 'test-domain prior values observed')
    new_text = new_text.replace('因子集合和权重在开发集DIOR→COCO确定', 'factor set and weights determined on DIOR→COCO dev set')
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.text = ''
    if cell.paragraphs and cell.paragraphs[0].runs:
        cell.paragraphs[0].runs[0].text = new_text
    else:
        cell.text = new_text
    print(f'Fixed Table 14, Row 8, Col 5: {new_text[:80]}')

# Verify no Chinese remains
remaining = 0
for t_idx, table in enumerate(doc.tables):
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            if re.search(r'[\u4e00-\u9fff]', cell.text):
                remaining += 1
                print(f'  STILL CN: Table {t_idx}, Row {r_idx}, Col {c_idx}: {cell.text[:60]}')

print(f'\nRemaining Chinese cells: {remaining}')

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')
print('Saved.')
