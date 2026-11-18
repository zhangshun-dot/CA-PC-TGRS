# -*- coding: utf-8 -*-
r"""update_ppt_experiments.py — 补充实验数据到PPT
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from copy import deepcopy

PPT_PATH = r'C:\Users\张三\Downloads\A-CSPC论文汇报.pptx'
OUTPUT_PATH = r'C:\Users\张三\Downloads\A-CSPC论文汇报_实验补充版.pptx'

prs = Presentation(PPT_PATH)

print(f'原始PPT: {len(prs.slides)}张幻灯片')

# 颜色定义
DARK_BLUE = RGBColor(0x1F, 0x3A, 0x5F)
MEDIUM_BLUE = RGBColor(0x2E, 0x5C, 0x8A)
LIGHT_BLUE = RGBColor(0xE8, 0xF0, 0xF8)
ACCENT_GREEN = RGBColor(0x2E, 0x7D, 0x32)
ACCENT_RED = RGBColor(0xC6, 0x28, 0x28)
ACCENT_ORANGE = RGBColor(0xE6, 0x7E, 0x22)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
LIGHT_GRAY = RGBColor(0xF5, 0xF5, 0xF5)

def add_slide_with_title(prs, title_text, page_num):
    """添加一张带标题的幻灯片，使用空白布局"""
    # 使用空白布局
    blank_layout = prs.slide_layouts[6]  # 通常是空白布局
    slide = prs.slides.add_slide(blank_layout)
    
    # 添加标题栏背景
    title_bg = slide.shapes.add_shape(
        1,  # rectangle
        Inches(0), Inches(0), Inches(13.33), Inches(0.8)
    )
    title_bg.fill.solid()
    title_bg.fill.fore_color.rgb = DARK_BLUE
    title_bg.line.fill.background()
    
    # 添加标题文字
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(11), Inches(0.6))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    # 添加页码
    page_box = slide.shapes.add_textbox(Inches(12), Inches(0.2), Inches(1), Inches(0.4))
    tf = page_box.text_frame
    p = tf.paragraphs[0]
    p.text = f'{page_num} / 16'
    p.font.size = Pt(12)
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.RIGHT
    
    return slide

def add_text_box(slide, left, top, width, height, text, font_size=14, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.LEFT):
    """添加文本框"""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = alignment
    return box

def add_table(slide, left, top, width, height, data, header_color=MEDIUM_BLUE):
    """添加表格"""
    rows = len(data)
    cols = len(data[0])
    table_shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height))
    table = table_shape.table
    
    for i, row_data in enumerate(data):
        for j, cell_text in enumerate(row_data):
            cell = table.cell(i, j)
            cell.text = str(cell_text)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(11)
                paragraph.alignment = PP_ALIGN.CENTER
                if i == 0:  # 表头
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = WHITE
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = header_color
                else:
                    paragraph.font.color.rgb = DARK_GRAY
                    if i % 2 == 0:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = LIGHT_GRAY
    
    return table_shape

print('\n=== 添加实验幻灯片 ===')

# 幻灯片1: SOTA对比
print('\n1. 添加SOTA对比幻灯片')
slide_sota = add_slide_with_title(prs, '实验：SOTA对比（目标域无标注主对比）', 10)

# 左侧：SOTA对比表
sota_data = [
    ['方法', 'AP50', '增益', '信息预算'],
    ['Baseline (GDINO-tiny)', '0.3707', '—', '无'],
    ['CORAL (相关性对齐)', '0.3917', '+0.0210', '无标签候选框'],
    ['DANN (域对抗, 5种子)', '0.3156±0.109', '−0.0551', '200张无标签图训练'],
    ['A-CSPC 固定参数', '0.3873', '+0.0166', '开发域选参'],
    ['A-CSPC 自适应先验', '0.4168', '+0.0461', '无目标域标注'],
    ['A-CSPC 主配置 (上界)', '0.4353', '+0.0646', '目标域GT调参'],
]
add_table(slide_sota, 0.5, 1.2, 7.5, 3.5, sota_data)

# 右侧：关键发现
add_text_box(slide_sota, 8.3, 1.2, 4.5, 0.4, '关键发现', font_size=18, bold=True, color=DARK_BLUE)
findings = [
    '• A-CSPC自适应版+0.0461是CORAL的2.20倍',
    '• 零训练参数，输出确定性（DANN需训1.3M参数，种子间sd=0.109）',
    '• 固定参数版+0.0166略低于CORAL+0.0210（如实披露）',
    '• DANN在冻结VL特征+候选框级适配下无收益，CI跨0',
    '• 主配置(+0.0646)使用目标域GT调参，作为上界参考',
]
for i, finding in enumerate(findings):
    add_text_box(slide_sota, 8.3, 1.7 + i*0.5, 4.5, 0.5, finding, font_size=12)

# 底部：协议说明
add_text_box(slide_sota, 0.5, 5.0, 12, 0.4, 
    '协议说明：表中前5行均不使用目标域标注（主对比）；最后一行为使用目标域GT调参的上界参考，不参与主对比。',
    font_size=11, color=DARK_GRAY)

# 幻灯片2: VOC→COCO 20类扩类别验证
print('\n2. 添加VOC→COCO 20类扩类别验证幻灯片')
slide_voc = add_slide_with_title(prs, '实验：VOC→COCO 20类扩类别验证', 11)

# 左侧：核心结果
add_text_box(slide_voc, 0.5, 1.2, 6, 0.4, '核心结果', font_size=18, bold=True, color=DARK_BLUE)
voc_results = [
    '• 固定参数(β=1.0,λ=0.25)：均值增益+0.0301',
    '  19/20类正增益',
    '• 自适应先验(α=1.5)：均值增益+0.0328',
    '  18/20类正增益，15/20类优于固定参数',
    '• 亮点类别：',
    '  person: 自适应+0.0319 vs 固定+0.0203',
    '  potted plant: 自适应−0.0052 vs 固定−0.0298（挽救负增益）',
    '  bus: 自适应+0.0386 vs 固定+0.0286',
]
for i, result in enumerate(voc_results):
    add_text_box(slide_voc, 0.5, 1.7 + i*0.4, 6, 0.4, result, font_size=12)

# 右侧：类别覆盖
add_text_box(slide_voc, 7, 1.2, 5.8, 0.4, '类别覆盖（20类）', font_size=18, bold=True, color=DARK_BLUE)
categories = [
    '交通工具(7): airplane, bicycle, boat, bus, car, motorcycle, train',
    '动物(6): cat, dog, horse, cow, sheep, bird',
    '室内物体(6): bottle, chair, dining table, couch, potted plant, tv',
    '人物(1): person',
]
for i, cat in enumerate(categories):
    add_text_box(slide_voc, 7, 1.7 + i*0.6, 5.8, 0.6, cat, font_size=12)

# 底部：意义
add_text_box(slide_voc, 0.5, 5.5, 12, 0.8,
    '意义：验证了方法在自然图像源域（VOC）、更多类别（20类）、更细粒度类别上的普适性，不依赖DIOR遥感域的特殊结构。物理先验驱动的自适应控制在跨域设定上稳定有效。',
    font_size=13, bold=True, color=DARK_BLUE)

# 幻灯片3: 统计显著性与效率分析
print('\n3. 添加统计显著性与效率分析幻灯片')
slide_stats = add_slide_with_title(prs, '实验：统计显著性检验与效率分析', 12)

# 左侧：Bootstrap置信区间
add_text_box(slide_stats, 0.5, 1.2, 6, 0.4, 'Bootstrap统计显著性检验', font_size=18, bold=True, color=DARK_BLUE)
bootstrap_data = [
    ['指标', '数值'],
    ['重采样次数', '1000次 (seed=42)'],
    ['测试图像', '90张 (配对图像级)'],
    ['均值增益', '+0.0668'],
    ['95%置信区间', '[0.052, 0.082]'],
    ['p值 (单侧)', '0.001'],
    ['正增益次数', '999/1000'],
    ['NWPU复现p值', '0.001'],
]
add_table(slide_stats, 0.5, 1.7, 6, 3.2, bootstrap_data)

# 右侧：效率分析
add_text_box(slide_stats, 7, 1.2, 5.8, 0.4, '效率与参数量分析', font_size=18, bold=True, color=DARK_BLUE)
efficiency_data = [
    ['方法', '可学习参数', '训练'],
    ['A-CSPC (本文)', '0', '零训练'],
    ['CoOp 提示学习', '4,096', '10 epochs'],
    ['CLIP-Adapter', '525,312', '20 epochs'],
]
add_table(slide_stats, 7, 1.7, 5.8, 1.8, efficiency_data)

# 推理时间
add_text_box(slide_stats, 7, 3.8, 5.8, 0.4, '推理时间分解 (RTX 5060)', font_size=14, bold=True, color=DARK_BLUE)
time_text = [
    '• GDINO-tiny检测: 734.2 ms/图',
    '• 候选框CLIP编码: 16.9 ms/框 (≈355 ms/图)',
    '• A-CSPC打分: 0.0001 ms/框 (可忽略)',
    '• 端到端: ≈1090 ms/图 (≈1.5×检测时间)',
]
for i, t in enumerate(time_text):
    add_text_box(slide_stats, 7, 4.3 + i*0.35, 5.8, 0.35, t, font_size=11)

# 底部：意义
add_text_box(slide_stats, 0.5, 5.5, 12, 0.6,
    '意义：Bootstrap检验证明增益统计显著（p=0.001），非随机波动；零参数+快推理使方法特别适合算力受限或无法访问目标域训练数据的场景。',
    font_size=13, bold=True, color=DARK_BLUE)

# 幻灯片4: 敏感性分析
print('\n4. 添加敏感性分析幻灯片')
slide_sens = add_slide_with_title(prs, '实验：敏感性分析（预过滤阈值与源域样本量）', 13)

# 左侧：预过滤阈值敏感性
add_text_box(slide_sens, 0.5, 1.2, 6, 0.4, '预过滤阈值敏感性', font_size=18, bold=True, color=DARK_BLUE)
prefilter_data = [
    ['阈值', '6类均值增益', '评价'],
    ['0.01', '+0.0615', '候选框多，基线低，增益大'],
    ['0.03', '+0.0567', '主配置（平衡）'],
    ['0.05', '+0.0517', '候选框少，基线高，增益小'],
]
add_table(slide_sens, 0.5, 1.7, 6, 1.8, prefilter_data)

add_text_box(slide_sens, 0.5, 3.7, 6, 1.0,
    '结论：方法对预过滤阈值不敏感，在0.01-0.05范围内均保持显著正增益。阈值降低时绝对增益反而增大，说明方法在低质量候选框池上校正空间更大。',
    font_size=12)

# 右侧：源域样本量敏感性
add_text_box(slide_sens, 7, 1.2, 5.8, 0.4, '源域样本量敏感性', font_size=18, bold=True, color=DARK_BLUE)
srcsize_data = [
    ['每类样本数', '6类均值增益', '差异'],
    ['5', '+0.0647', '基准'],
    ['10', '+0.0641', '−0.0006'],
    ['20', '+0.0652', '+0.0005'],
]
add_table(slide_sens, 7, 1.7, 5.8, 1.8, srcsize_data)

add_text_box(slide_sens, 7, 3.7, 5.8, 1.0,
    '结论：方法对源域标注数据量需求极低，仅需每类5个样本即可获得与20个样本相当的增益（差异<0.001）。特别适用于源域标注稀缺的跨域场景。',
    font_size=12)

# 底部：标准COCO指标
add_text_box(slide_sens, 0.5, 5.3, 12, 0.8,
    '补充：标准COCO指标AP@[0.5:0.95] — A-CSPC取得+0.0463增益，6/6类正增益（airplane +0.0129、boat +0.0010、car +0.0460、truck +0.0732、bus +0.0670、motorcycle +0.0779），证明方法在更严格的评估指标下同样有效。',
    font_size=12, bold=True, color=DARK_BLUE)

print(f'\n修改后PPT: {len(prs.slides)}张幻灯片')

prs.save(OUTPUT_PATH)
print(f'\nPPT已保存: {OUTPUT_PATH}')
print('实验数据补充完成')
