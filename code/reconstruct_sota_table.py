# -*- coding: utf-8 -*-
r"""reconstruct_sota_table.py — 重构SOTA对比表
拆成两组：
  (a) 目标域无标注（主对比）：Baseline / B2 / T4 / CORAL / DANN(200张) / A-CSPC固定 / A-CSPC自适应
  (b) 使用目标域GT（上界参考）：A-CSPC主配置(2.0,0.5) +0.0646
同时更新正文段落，加入DANN训练数据量扫描结论和T4净增量分析。
"""
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt
from lxml import etree

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v15.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v16.docx'

doc = Document(PAPER_PATH)

# 找到原来的SOTA对比段落和表格，删除后重新插入
# 先找到SOTA段落的位置
sota_para_idx = None
for i, p in enumerate(doc.paragraphs):
    if '与现有域自适应重排序方法的对比' in p.text and '相同候选框池' in p.text:
        sota_para_idx = i
        break

print(f'原SOTA段落在段落{sota_para_idx}')

if sota_para_idx is None:
    print('未找到原SOTA段落')
    exit(1)

# 删除原来的SOTA段落、表格、表注、无目标域段落、表格、表注
# 从sota_para_idx开始，删除直到下一个章节标题
elements_to_remove = []
sota_elem = doc.paragraphs[sota_para_idx]._element
current = sota_elem
while current is not None:
    # 检查是否是下一个章节标题
    text = ''.join(t.text or '' for t in current.iter(qn('w:t')))
    if text.strip().startswith('4.3') or text.strip().startswith('4.4') or text.strip().startswith('4.5'):
        break
    elements_to_remove.append(current)
    current = current.getnext()

print(f'要删除的元素数: {len(elements_to_remove)}')
for elem in elements_to_remove:
    elem.getparent().remove(elem)

# 在sota_para_idx的位置（现在是下一个段落之前）插入新的SOTA内容
# 找到插入位置：sota_para_idx之前的元素
insert_before = doc.paragraphs[sota_para_idx]._element if sota_para_idx < len(doc.paragraphs) else None

def insert_paragraph_before(elem, text):
    """在指定元素之前插入段落"""
    new_p = etree.SubElement(elem.getparent(), qn('w:p'))
    elem.addprevious(new_p)
    run = etree.SubElement(new_p, qn('w:r'))
    t = etree.SubElement(run, qn('w:t'))
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    return new_p

def insert_table_before(elem, data):
    """在指定元素之前插入表格"""
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    # 手动设置边框
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else etree.SubElement(tbl, qn('w:tblPr'))
    borders = etree.SubElement(tblPr, qn('w:tblBorders'))
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = etree.SubElement(borders, qn(f'w:{border_name}'))
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), '000000')
    # 填充数据
    for i, row_data in enumerate(data):
        row = table.rows[i]
        for j, cell_text in enumerate(row_data):
            cell = row.cells[j]
            cell.text = cell_text
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    if i == 0:
                        run.font.bold = True
    # 移动表格到指定位置
    table_elem = table._element
    table_elem.getparent().remove(table_elem)
    elem.addprevious(table_elem)
    return table_elem

# ══════════════════════════ 插入新的SOTA内容 ══════════════════════════

# 主对比段落
main_text = (
    '与现有域自适应重排序方法的对比（相同候选框池、相同CLIP特征、公平协议）。'
    '为避免信息预算不一致导致的不公平对比，结果按"是否使用目标域标注"分为两组：'
    '表SOTA-a为目标域无标注设置下的主对比（所有方法均不使用目标域GT），'
    '表SOTA-b为使用目标域GT调参的上界参考，不参与主对比。'
)
insert_paragraph_before(insert_before, main_text)

# 表SOTA-a：目标域无标注（主对比）
table_a_data = [
    ['方法', '目标域数据', 'airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle', '6类平均', '增益'],
    ['Baseline (GDINO-tiny)', '无', '0.7484', '0.1758', '0.1245', '0.2345', '0.5317', '0.4092', '0.3707', '—'],
    ['B2 源域原型', '无', '0.7579', '0.1650', '0.1063', '0.2498', '0.4909', '0.4357', '0.3676', '−0.0031'],
    ['T4 纯文本原型', '无', '0.7637', '0.1904', '0.1466', '0.2770', '0.5733', '0.4876', '0.4064', '+0.0357'],
    ['CORAL 相关性对齐', '无标签候选框', '0.7495', '0.1898', '0.1268', '0.2653', '0.5774', '0.4415', '0.3917', '+0.0210'],
    ['DANN 域对抗 (200张图训练)', '无标签候选框', '0.7475', '0.1816', '0.1245', '0.2352', '0.5317', '0.4092', '0.3716', '+0.0009'],
    ['A-CSPC 固定参数 (β=1.0,λ=0.25)', '无', '0.7598', '0.1659', '0.1270', '0.2614', '0.5326', '0.4775', '0.3873', '+0.0166'],
    ['A-CSPC 物理先验自适应 (α=1.5)', '无', '—', '—', '—', '—', '—', '—', '0.4185', '+0.0478'],
]
insert_table_before(insert_before, table_a_data)

# 表SOTA-a表注
caption_a = '表SOTA-a　目标域无标注设置下的主对比（所有方法均不使用目标域GT；DANN使用200张目标域图的无标签候选框特征训练，CORAL使用无标签候选框特征计算协方差；AP50，11点插值）'
insert_paragraph_before(insert_before, caption_a)

# DANN训练数据量扫描分析
dann_analysis_text = (
    'DANN训练数据量扫描分析：为排除"域对抗失败是因训练数据不足"的质疑，'
    '额外扫描了目标域训练图数从10张到200张的四个规模（10/50/100/200张），'
    '6类平均增益分别为−0.0958、−0.0087、−0.0049、+0.0009。'
    '10张图时严重欠训练/过拟合，随着数据量增加性能逐渐恢复到基线水平，'
    '但即使在200张图训练下增益也仅+0.0009，基本等于基线。'
    '这表明域对抗训练在冻结CLIP特征+候选框级适配的设置下确实无法带来增益，'
    '而非训练数据不足导致的稻草人baseline。'
)
insert_paragraph_before(insert_before, dann_analysis_text)

# T4净增量分析
t4_analysis_text = (
    '净增量分析：A-CSPC主配置(+0.0646)与纯文本原型T4(+0.0357)的差值为+0.0289，'
    '即非对称原型构造的净增量占总增益的44.7%，其余55.3%来自"用目标域文本嵌入重排序"这一简单操作。'
    '这一分解如实披露了方法各组件的贡献，避免过度宣称非对称构造的作用。'
    '在目标域无标注设置下，A-CSPC物理先验自适应(+0.0478)相比T4(+0.0357)的净增量为+0.0121，'
    '说明物理先验驱动的自适应参数控制在无标注设置下仍能带来稳定增益。'
)
insert_paragraph_before(insert_before, t4_analysis_text)

# 表SOTA-b：使用目标域GT（上界参考）
table_b_data = [
    ['方法', '目标域数据', '6类平均AP50', '增益', '说明'],
    ['A-CSPC 主配置 (β=2.0,λ=0.5)', 'GT调参', '0.4353', '+0.0646', '上界参考，不参与主对比'],
]
insert_table_before(insert_before, table_b_data)

# 表SOTA-b表注
caption_b = '表SOTA-b　使用目标域GT调参的上界参考（A-CSPC主配置使用COCO GT进行β/λ网格搜索，仅作为性能上界参考，不参与表SOTA-a的主对比；AP50，11点插值）'
insert_paragraph_before(insert_before, caption_b)

# 保存
doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('SOTA对比表已重构为两组（主对比+上界参考）')
print('已加入DANN训练数据量扫描分析和T4净增量分析')
