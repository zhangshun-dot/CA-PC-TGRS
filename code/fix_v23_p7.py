# -*- coding: utf-8 -*-
r"""fix_v23_p7.py — 表10-18 → 表8-16，实现全连续编号
表10→表8（类别映射）
表11→表9（三源域复现）
表12→表10（负原型构建）
表13→表11（检测器谱系）
表14→表12（多目标域对比）
表15→表13（反向迁移）
表16→表14（信息使用审计）
表17→表15（物理先验因子）
表18→表16（LODO）
"""
from docx import Document
from docx.oxml.ns import qn
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'

doc = Document(PAPER_PATH)

# 表号映射（从大到小修改，避免冲突）
table_map = {
    18: 16,  # LODO
    17: 15,  # 物理先验因子
    16: 14,  # 信息使用审计
    15: 13,  # 反向迁移
    14: 12,  # 多目标域对比
    13: 11,  # 检测器谱系
    12: 10,  # 负原型构建
    11: 9,   # 三源域复现
    10: 8,   # 类别映射
}

print('修改表号：表10-18 → 表8-16')

# 先修改题注
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    for old_num, new_num in table_map.items():
        # 匹配题注开头的"表X"
        if text.startswith(f'表{old_num}') or text.startswith(f'表{old_num}（'):
            old_text = p.text
            # 只替换开头的表号
            new_text = re.sub(rf'^表{old_num}', f'表{new_num}', old_text)
            for run in p.runs:
                run.text = ''
            if p.runs:
                p.runs[0].text = new_text
            else:
                p.add_run(new_text)
            print(f'  段落{i}: 表{old_num}→表{new_num}（题注）')
            break

# 再修改正文中的引用
print('\n修改正文中的表引用：')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    modified = False
    for old_num, new_num in table_map.items():
        # 匹配正文中的"表X"（不是题注开头）
        # 使用负向断言，确保不是"表X0"等
        pattern = rf'表{old_num}(?!\d)'
        if re.search(pattern, text):
            # 排除题注本身（已经修改过了）
            if not p.text.strip().startswith(f'表{new_num}'):
                text = re.sub(pattern, f'表{new_num}', text)
                modified = True
    if modified:
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = text
        else:
            p.add_run(text)
        print(f'  段落{i}: 引用已更新')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('表号全连续修复完成')
