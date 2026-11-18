from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print("=== SECTION 2: RELATED WORK (paras 16-27) ===")
for i in range(16, 28):
    p = doc.paragraphs[i]
    if p.text.strip():
        print(f"[{i}] ({p.style.name})")
        print(p.text[:500])
        print()

print("=== SECTION 3: METHODS (paras 28-41) ===")
for i in range(28, 42):
    p = doc.paragraphs[i]
    if p.text.strip():
        print(f"[{i}] ({p.style.name})")
        print(p.text[:500])
        print()
