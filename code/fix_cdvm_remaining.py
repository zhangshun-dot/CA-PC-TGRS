# -*- coding: utf-8 -*-
from docx import Document
doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v9.docx')

def replace_in_para(idx, old, new):
    p = doc.paragraphs[idx]
    if old in p.text:
        for run in p.runs:
            if old in run.text:
                run.text = run.text.replace(old, new)
                return True
        full = p.text.replace(old, new)
        p.runs[0].text = full
        for run in p.runs[1:]:
            run.text = ''
        return True
    return False

r1 = replace_in_para(43, 'CDVM[1]（全量六类口径）−0.071', '源域视觉原型基线（β=0，全量六类口径）−0.071')
print('段落43:', '成功' if r1 else '失败')

r2 = replace_in_para(56, 'CDVM（N=100口径）−0.0451', '源域视觉原型基线（β=0，N=100口径）−0.0451')
print('段落56:', '成功' if r2 else '失败')

r3 = replace_in_para(186, '旧 CDVM 对照线', '旧消融对照线')
print('段落186:', '成功' if r3 else '失败')

r4 = replace_in_para(209, '旧 CDVM 对照线', '旧消融对照线')
print('段落209:', '成功' if r4 else '失败')

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v9.docx')
print('已保存')

doc2 = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v9.docx')
count = sum(1 for p in doc2.paragraphs if 'CDVM' in p.text)
print('验证：CDVM残留', count, '处')
