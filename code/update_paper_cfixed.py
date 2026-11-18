# -*- coding: utf-8 -*-
r"""update_paper_cfixed.py — 批量更新论文中的C_fixed实验结果

更新内容：
  1. §4.2 主结果：+0.0646 → +0.0714，逐类别更新
  2. §4.4 消融实验：更新各配置增益
  3. §4.8 多检测器：GDINO-tiny/base/OWLv2更新
  4. §4.9 β/λ扫描：更新数值
  5. 摘要、贡献点、结论中的主结果数值
"""
import sys
from docx import Document
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v2.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v3.docx'

doc = Document(PAPER_PATH)

# 统计更新次数
updates = []

def update_paragraph(idx, old_text, new_text, reason):
    """更新段落中的文本"""
    p = doc.paragraphs[idx]
    if old_text in p.text:
        # 保留格式，只替换文本
        for run in p.runs:
            if old_text in run.text:
                run.text = run.text.replace(old_text, new_text)
                updates.append(f'[{idx}] {reason}: "{old_text}" → "{new_text}"')
                return True
        # 如果跨run，用整段替换
        full_text = p.text
        if old_text in full_text:
            new_full = full_text.replace(old_text, new_text)
            # 清空所有run，写入第一个run
            if p.runs:
                p.runs[0].text = new_full
                for run in p.runs[1:]:
                    run.text = ''
            updates.append(f'[{idx}] {reason} (跨run): "{old_text[:30]}..." → "{new_text[:30]}..."')
            return True
    return False

# ═══════════════════════════════════════════════════════════════
# 1. §4.2 主结果（第43段）
# ═══════════════════════════════════════════════════════════════
print('[1/5] 更新§4.2主结果...')

# 主结果均值
update_paragraph(43, '+0.0646的AP50增益', '+0.0714的AP50增益', '主结果均值')

# 逐类别增益
update_paragraph(43, 'airplane+0.0117', 'airplane+0.0228', 'airplane增益')
update_paragraph(43, 'boat+0.0253', 'boat+0.0316', 'boat增益')
update_paragraph(43, 'car+0.0809', 'car+0.1310', 'car增益')
update_paragraph(43, 'truck+0.0774', 'truck+0.0796', 'truck增益')
update_paragraph(43, 'bus+0.0735', 'bus+0.0639', 'bus增益')
update_paragraph(43, 'motorcycle+0.1189', 'motorcycle+0.0992', 'motorcycle增益')

# ═══════════════════════════════════════════════════════════════
# 2. §4.4 消融实验（第56段）
# ═══════════════════════════════════════════════════════════════
print('[2/5] 更新§4.4消融实验...')

# 原消融文本："增益从+0.0652降至−0.0005"
# 新消融：完整A-CSPC +0.0714，去外推 -0.0920
update_paragraph(56, '增益从+0.0652降至−0.0005', '增益从+0.0714降至−0.0920', '去外推消融')

# 原："纯外推增益+0.0239（占总增益37%）"
# 新：去负原型(λ=0)增益-0.0384
update_paragraph(56, '纯外推增益+0.0239（占总增益37%）', '去负原型配置增益−0.0384，说明负原型抑制是必要组件', '去负原型消融')

# ═══════════════════════════════════════════════════════════════
# 3. §4.8 多检测器（第74段）
# ═══════════════════════════════════════════════════════════════
print('[3/5] 更新§4.8多检测器...')

update_paragraph(74, 'GDINO-tiny（基线0.3707）增益+0.0646', 'GDINO-tiny（基线0.3707）增益+0.0714', 'GDINO-tiny增益')
update_paragraph(74, 'GDINO-base（基线0.4037）增益+0.0790', 'GDINO-base（基线0.4037）增益+0.0841', 'GDINO-base增益')
update_paragraph(74, 'OWLv2（基线0.6800）增益−0.0316', 'OWLv2（基线0.6800）增益−0.0378', 'OWLv2增益')

# 图4标题（第76段）
update_paragraph(76, 'GDINO-tiny +0.0646、GDINO-base +0.0790', 'GDINO-tiny +0.0714、GDINO-base +0.0841', '图4标题')

# ═══════════════════════════════════════════════════════════════
# 4. §4.9 β/λ扫描（第80段）
# ═══════════════════════════════════════════════════════════════
print('[4/5] 更新§4.9 β/λ扫描...')

update_paragraph(80, 'β=1.0时+0.0318，β=4.0时−0.0684进入崩溃区', 'β=1.0时+0.0105，β=4.0时−0.0461进入崩溃区', 'β扫描')

# ═══════════════════════════════════════════════════════════════
# 5. 摘要、贡献点、结论中的主结果数值
# ═══════════════════════════════════════════════════════════════
print('[5/5] 更新摘要、贡献点、结论...')

# 搜索全文中的+0.0646（主结果）
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if '+0.0646' in text and i != 43:  # 第43段已经更新
        update_paragraph(i, '+0.0646', '+0.0714', f'段落{i}主结果均值')

# 搜索全文中的0.0646（不带+号）
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if '0.0646' in text and '0.0714' not in text:
        update_paragraph(i, '0.0646', '0.0714', f'段落{i}主结果数值')

# ═══════════════════════════════════════════════════════════════
# 6. 新增§3.5 校准加权融合策略（跳过，手动添加）
# ═══════════════════════════════════════════════════════════════
print('[6/6] 跳过新增§3.5（后续手动添加）...')

# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════
doc.save(OUTPUT_PATH)

print(f'\n更新完成！共{len(updates)}处更新：')
for u in updates:
    print(f'  {u}')
print(f'\n保存至: {OUTPUT_PATH}')
