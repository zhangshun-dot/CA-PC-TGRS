from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print("=== FULL ABSTRACT ===")
for i in range(0, 6):
    p = doc.paragraphs[i]
    print(f"[{i}] ({p.style.name})")
    print(p.text)
    print()

print("=== INTRO (paras 5-20) ===")
for i in range(5, 20):
    p = doc.paragraphs[i]
    if p.text.strip():
        print(f"[{i}] ({p.style.name})")
        print(p.text[:400])
        print()
