from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print("=== SECTION 4: EXPERIMENTS (paras 41-80) ===")
for i in range(41, 81):
    p = doc.paragraphs[i]
    if p.text.strip():
        print(f"[{i}] ({p.style.name})")
        print(p.text[:400])
        print()
