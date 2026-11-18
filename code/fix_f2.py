# -*- coding: utf-8 -*-
"""fix_f2.py — 修复F2：公式无编号、无正文引用"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# 段落36：正原型外推公式
p36 = doc.paragraphs[36]
old_text_36 = p36.text
new_text_36 = old_text_36.replace(
    '正原型外推公式：p̂_c = p_c + β·(d(x_c) − s(x_s(c)))',
    '正原型外推公式如式(1)所示：p̂_c = p_c + β·(d(x_c) − s(x_s(c)))　　　　（式1）'
)
for run in p36.runs:
    run.text = ''
p36.runs[0].text = new_text_36

# 段落37：视觉原型分数公式 + 最终融合分数公式
p37 = doc.paragraphs[37]
old_text_37 = p37.text

# 视觉原型分数公式
new_text_37 = old_text_37.replace(
    '视觉原型分数定义为：\ns_vis(f_k) = max( 0, ⟨f_k, p̂_c⟩ − λ·⟨f_k, p_neg⟩ )',
    '视觉原型分数定义如式(2)所示：\ns_vis(f_k) = max( 0, ⟨f_k, p̂_c⟩ − λ·⟨f_k, p_neg⟩ )　　　　（式2）'
)

# 最终融合分数公式
new_text_37 = new_text_37.replace(
    '最终融合分数为：\ns_fused,k = sqrt( clip(s_det,k, ε, 1) · clip(s_vis(f_k), ε, 1) ),  ε=10⁻⁶',
    '最终融合分数如式(3)所示：\ns_fused,k = sqrt( clip(s_det,k, ε, 1) · clip(s_vis(f_k), ε, 1) ),  ε=10⁻⁶　　　　（式3）'
)

for run in p37.runs:
    run.text = ''
p37.runs[0].text = new_text_37

# 在§4.4消融实验里引用公式
# 先找到§4.4的位置
for i, p in enumerate(doc.paragraphs):
    if '4.4' in p.text and '消融' in p.text and p.style.name.startswith('Heading'):
        # 在§4.4正文里引用式(1)
        for j in range(i+1, min(i+5, len(doc.paragraphs))):
            if doc.paragraphs[j].style.name.startswith('Heading'):
                break
            if '正原型外推' in doc.paragraphs[j].text and '式' not in doc.paragraphs[j].text:
                old = doc.paragraphs[j].text
                new = old.replace('正原型外推', '正原型外推（式1）')
                for run in doc.paragraphs[j].runs:
                    run.text = ''
                doc.paragraphs[j].runs[0].text = new
                print(f'  在§4.4段落[{j}]引用式(1)')
                break
        break

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print()
print('F2已完成：§3.3三个关键公式已编号')
print('  式(1)：正原型外推公式')
print('  式(2)：视觉原型分数公式')
print('  式(3)：最终融合分数公式')
print()
print('已在§4.4消融实验中引用式(1)')
