from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')

print('=== Updated Structure (Sections 4.5 - 7) ===')
for i in range(70, 162):
    p = doc.paragraphs[i]
    text = p.text.strip()
    if text and (text.startswith(('4.', '5.', '6.', '7.', '8.')) or 'Supplementary' in text):
        if len(text) < 120:
            print(f'  [{i}] {text}')

print()
print('=== Content Verification ===')
print(f'4.5 content length: {len(doc.paragraphs[72].text)} chars')
print(f'4.5 starts with: {doc.paragraphs[72].text[:80]}...')

# Check Chinese remaining
para_chinese = sum(1 for p in doc.paragraphs if re.search(r'[\u4e00-\u9fff]', p.text))
table_chinese = 0
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            if re.search(r'[\u4e00-\u9fff]', cell.text):
                table_chinese += 1
print(f'Chinese remaining: {para_chinese} paragraphs, {table_chinese} table cells')

# Check key positive results are present
full_text = ' '.join(p.text for p in doc.paragraphs)
print(f'+0.0646 mentioned: {"+0.0646" in full_text}')
print(f'+0.0461 mentioned: {"+0.0461" in full_text}')
print(f'75% mentioned: {"75%" in full_text}')
print(f'24 scene classes mentioned: {"24 scene classes" in full_text or "24 scene class" in full_text}')
print(f'p=0.001 mentioned: {"p=0.001" in full_text}')
print(f'6/6 positive mentioned: {"6/6" in full_text}')
