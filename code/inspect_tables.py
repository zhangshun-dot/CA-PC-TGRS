from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')

print(f'Total tables: {len(doc.tables)}')
print()

for t_idx, table in enumerate(doc.tables):
    print(f'=== Table {t_idx} ({len(table.rows)} rows x {len(table.columns)} cols) ===')
    for r_idx, row in enumerate(table.rows):
        cells_text = []
        has_chinese = False
        for c_idx, cell in enumerate(row.cells):
            text = cell.text.strip()
            if re.search(r'[\u4e00-\u9fff]', text):
                cells_text.append(f'[CN:{text[:40]}]')
                has_chinese = True
            elif text:
                cells_text.append(text[:40])
        if has_chinese:
            sep = ' | '
            print(f'  Row {r_idx}: {sep.join(cells_text)}')
    print()
