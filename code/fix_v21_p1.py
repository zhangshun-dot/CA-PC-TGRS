# -*- coding: utf-8 -*-
r"""fix_v21_p1.py — 修复v20第一优先级的简单确定性项
1. 剥离参考文献[49]中的污染正文（段191）
2. 修正NWPU 5/6正增益矛盾（段68）
3. 检查表3显著性检验引用的具体位置
"""
from docx import Document
from docx.oxml.ns import qn

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v20.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v21.docx'

doc = Document(PAPER_PATH)

# ══════════════════════════ 修复1：参考文献[49]污染 ══════════════════════════
print('修复1：剥离参考文献[49]中的污染正文')
p191 = doc.paragraphs[191]
old_text = p191.text
print(f'  段落191长度: {len(old_text)}')
print(f'  开头: {old_text[:100]}')

# 找到污染正文的起始位置："配置选择探索性分析"
split_point = old_text.find('配置选择探索性分析')
if split_point > 0:
    # 保留[49]文献条目，删除污染正文
    clean_text = old_text[:split_point].strip()
    print(f'  分割点: {split_point}')
    print(f'  清理后: {clean_text[:150]}')
    print(f'  清理后长度: {len(clean_text)}')

    # 重新设置段落文本
    for run in p191.runs:
        run.text = ''
    if p191.runs:
        p191.runs[0].text = clean_text
    else:
        p191.add_run(clean_text)
    print('  参考文献[49]污染已剥离')
else:
    print('  未找到污染正文起始位置')

# ══════════════════════════ 修复2：NWPU 5/6矛盾 ══════════════════════════
print('\n修复2：修正NWPU 5/6正增益矛盾')
p68 = doc.paragraphs[68]
old_text = p68.text
if '5/6个映射类正增益' in old_text:
    new_text = old_text.replace('5/6个映射类正增益', '6/6个映射类正增益')
    for run in p68.runs:
        run.text = ''
    if p68.runs:
        p68.runs[0].text = new_text
    else:
        p68.add_run(new_text)
    print('  NWPU 5/6 → 6/6 已修正')
else:
    print('  未找到5/6表述')

# ══════════════════════════ 检查3：表3显著性检验引用 ══════════════════════════
print('\n检查3：表3显著性检验引用')
for i, p in enumerate(doc.paragraphs):
    if '表3' in p.text:
        print(f'  段落{i}: {p.text[:200]}')
        print()

# ══════════════════════════ 验证 ══════════════════════════
print('\n=== 验证 ===')
# 验证1
p191_new = doc.paragraphs[191].text
print(f'  段落191新长度: {len(p191_new)}')
print(f'  包含"配置选择探索性分析": {"配置选择探索性分析" in p191_new}')

# 验证2
p68_new = doc.paragraphs[68].text
print(f'  包含"5/6": {"5/6" in p68_new}')
print(f'  包含"6/6": {"6/6" in p68_new}')

# 保存
doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('v21第一阶段修复完成:')
print('  1. 参考文献[49]污染已剥离')
print('  2. NWPU 5/6 → 6/6 已修正')
