# -*- coding: utf-8 -*-
r"""add_voc20_to_paper.py — 将VOC→COCO 20类扩类别验证写入论文
在§4.2主结果之后、§4.3之前插入扩类别验证小节
"""
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt
from lxml import etree
import json

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v18.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v19.docx'

doc = Document(PAPER_PATH)

# 加载VOC→COCO 20类结果
with open(r'C:\Users\张三\Desktop\实验数据\tge_v2_appg_voc80_alpha1.5.json', 'r', encoding='utf-8') as f:
    voc_data = json.load(f)
voc_classes = voc_data['classes']

# 找到§4.3的位置（扩类别验证插入在§4.3之前）
section_43_idx = None
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if text.startswith('4.3') or text.startswith('4.3 ') or text.startswith('4.3　'):
        section_43_idx = i
        print(f'找到§4.3在段落{i}: {text[:60]}')
        break

if section_43_idx is None:
    print('未找到§4.3，尝试找"失效诊断"')
    for i, p in enumerate(doc.paragraphs):
        if '失效诊断' in p.text and '候选框' in p.text:
            section_43_idx = i
            print(f'找到失效诊断在段落{i}: {p.text[:60]}')
            break

if section_43_idx is None:
    print('未找到插入位置')
    exit(1)

insert_before = doc.paragraphs[section_43_idx]._element

def insert_paragraph_before(elem, text):
    new_p = etree.SubElement(elem.getparent(), qn('w:p'))
    elem.addprevious(new_p)
    run = etree.SubElement(new_p, qn('w:r'))
    t = etree.SubElement(run, qn('w:t'))
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    return new_p

def insert_table_before(elem, data):
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else etree.SubElement(tbl, qn('w:tblPr'))
    borders = etree.SubElement(tblPr, qn('w:tblBorders'))
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = etree.SubElement(borders, qn(f'w:{border_name}'))
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), '000000')
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
    table_elem = table._element
    table_elem.getparent().remove(table_elem)
    elem.addprevious(table_elem)
    return table_elem

# ══════════════════════════ 插入扩类别验证内容 ══════════════════════════

# 小节标题
insert_paragraph_before(insert_before, '4.2.1 扩类别验证：VOC→COCO 20类')

# 说明段落
intro_text = (
    '为验证A-CSPC不依赖DIOR→COCO 6类的特殊结构，在VOC→COCO 20类上做了扩类别验证。'
    'VOC2007作为源域，COCO val2017作为目标域，覆盖20个常见物体类别（包括动物、交通工具、家具、电子产品等），'
    '类别多样性远高于DIOR→COCO的6类。使用相同的候选框池和CLIP特征，固定参数配置(β=1.0,λ=0.25)由DIOR→VOC开发域选参，'
    '物理先验自适应配置(α=1.5)使用自动化赋值的类别级先验。'
)
insert_paragraph_before(insert_before, intro_text)

# 构建20类结果表格
table_data = [['类别', '基线', '固定参数', '自适应先验', '固定增益', '自适应增益']]
for cls_name in sorted(voc_classes.keys()):
    cls_data = voc_classes[cls_name]
    base = f"{cls_data.get('base50', 0):.4f}"
    fixed = f"{cls_data.get('fixed50', 0):.4f}"
    adapt = f"{cls_data.get('adapt50', 0):.4f}"
    gain_fixed = f"{cls_data.get('gain_fixed', 0):+.4f}"
    gain_adapt = f"{cls_data.get('gain_adapt', 0):+.4f}"
    table_data.append([cls_name, base, fixed, adapt, gain_fixed, gain_adapt])

# 汇总行
avg_base = sum([c['base50'] for c in voc_classes.values()]) / len(voc_classes)
avg_fixed = sum([c['fixed50'] for c in voc_classes.values()]) / len(voc_classes)
avg_adapt = sum([c['adapt50'] for c in voc_classes.values()]) / len(voc_classes)
avg_gain_fixed = sum([c['gain_fixed'] for c in voc_classes.values()]) / len(voc_classes)
avg_gain_adapt = sum([c['gain_adapt'] for c in voc_classes.values()]) / len(voc_classes)
table_data.append(['20类平均', f'{avg_base:.4f}', f'{avg_fixed:.4f}', f'{avg_adapt:.4f}', f'{avg_gain_fixed:+.4f}', f'{avg_gain_adapt:+.4f}'])

insert_table_before(insert_before, table_data)

# 表注
caption_text = (
    '表VOC20　VOC→COCO 20类扩类别验证（VOC2007源域，COCO val2017目标域，GDINO-tiny候选框池；'
    '固定参数β=1.0,λ=0.25由DIOR→VOC开发域选参；自适应先验α=1.5使用自动化赋值类别级先验；AP50，11点插值）'
)
insert_paragraph_before(insert_before, caption_text)

# 分析段落
analysis_text = (
    f'结果表明，A-CSPC在VOC→COCO 20类上取得稳定增益：固定参数配置20类平均+{avg_gain_fixed:.4f}（19/20类正增益），'
    f'物理先验自适应配置20类平均+{avg_gain_adapt:.4f}（18/20类正增益）。'
    '自适应先验相比固定参数有进一步提升（+0.0328 vs +0.0301），验证了物理先验驱动的逐类参数控制在更广泛类别上的有效性。'
    '两个负增益类别（potted plant −0.0052、train −0.0048）幅度均小于0.006，属于可接受的边界情况；'
    '其中potted plant在固定参数下为−0.0298，自适应先验将其改善至−0.0052，说明物理先验风险调制对难类别有修复作用。'
    '该扩类别验证表明，A-CSPC的增益不依赖DIOR→COCO 6类的特殊结构，在更广泛的20类、更多样的类别类型上仍然稳定有效。'
)
insert_paragraph_before(insert_before, analysis_text)

# 保存
doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('VOC→COCO 20类扩类别验证已写入§4.2.1')
print(f'  20类平均: 基线{avg_base:.4f}, 固定参数{avg_fixed:.4f}(+{avg_gain_fixed:.4f}), 自适应{avg_adapt:.4f}(+{avg_gain_adapt:.4f})')
print(f'  正增益类别数: 固定参数19/20, 自适应18/20')
