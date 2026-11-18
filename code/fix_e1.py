# -*- coding: utf-8 -*-
"""fix_e1.py — 修复E1：删除"如实披露"等自我标榜措辞"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# 需要替换的措辞对
replacements = [
    # (旧表述, 新表述)
    ('如实披露β=2.0/λ=0.5的目标域调参来源', '明确β=2.0/λ=0.5的目标域调参来源'),
    ('这一分解如实披露了方法各组件的贡献', '这一分解显示了方法各组件的贡献'),
    ('配置选择审计如实披露超参数来源', '配置选择审计明确超参数来源'),
    ('信息使用审计如实披露各配置的信息预算', '信息使用审计明确各配置的信息预算'),
    ('均在补充材料中完整披露', '均在补充材料中披露'),
    ('所有超参数来源与信息使用协议均如实披露，失败案例（boat转负、OWLv2负增益、反向迁移失败）作为方法边界的证据完整保留',
     '所有超参数来源与信息使用协议均已披露，失败案例（boat转负、OWLv2负增益、反向迁移失败）作为方法边界的证据保留'),
]

count = 0
for i, p in enumerate(doc.paragraphs):
    old_text = p.text
    new_text = old_text
    for old_phrase, new_phrase in replacements:
        if old_phrase in new_text:
            new_text = new_text.replace(old_phrase, new_phrase)
            count += 1
            print(f'  [{i}] 替换: "{old_phrase[:40]}..." → "{new_phrase[:40]}..."')
    if new_text != old_text:
        for run in p.runs:
            run.text = ''
        p.runs[0].text = new_text

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')
print(f'\nE1已完成：共替换{count}处自我标榜措辞')
