from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print(f"Total paragraphs: {len(doc.paragraphs)}")
print()

print("=== REMAINING SECTIONS (paras 81-239) ===")
for i in range(81, len(doc.paragraphs)):
    p = doc.paragraphs[i]
    if p.text.strip():
        style = p.style.name if p.style else 'None'
        # Only print headings and first 200 chars of body text
        if 'Heading' in style:
            print(f"\n[{i}] ({style}) {p.text}")
        else:
            print(f"[{i}] ({style}) {p.text[:250]}")
