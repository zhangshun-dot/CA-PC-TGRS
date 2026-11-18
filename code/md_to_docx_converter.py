# -*- coding: utf-8 -*-
"""将Markdown论文框架转换为格式规范的docx文档"""
import re
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

def set_font(run, font_name='宋体', font_size=12, bold=False, color=None):
    """设置字体"""
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color
    # 设置中文字体
    r = run._element
    rPr = r.find(qn('w:rPr'))
    if rPr is None:
        rPr = r.makeelement(qn('w:rPr'), {})
        r.insert(0, rPr)
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = rPr.makeelement(qn('w:rFonts'), {})
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), font_name)

def add_heading(doc, text, level):
    """添加标题"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    if level == 1:
        set_font(run, '黑体', 18, bold=True)
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(12)
    elif level == 2:
        set_font(run, '黑体', 15, bold=True)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(8)
    elif level == 3:
        set_font(run, '黑体', 13, bold=True)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(6)
    elif level == 4:
        set_font(run, '黑体', 12, bold=True)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
    return p

def add_paragraph(doc, text, bold=False, indent=True):
    """添加正文段落"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(6)
    # 处理加粗
    parts = re.split(r'\*\*(.+?)\*\*', text)
    for i, part in enumerate(parts):
        if i % 2 == 1:
            run = p.add_run(part)
            set_font(run, '宋体', 12, bold=True)
        else:
            run = p.add_run(part)
            set_font(run, '宋体', 12, bold=bold)
    return p

def add_list_item(doc, text, level=0):
    """添加列表项"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Pt(24 + level * 24)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(3)
    # 处理加粗
    parts = re.split(r'\*\*(.+?)\*\*', text)
    for i, part in enumerate(parts):
        if i % 2 == 1:
            run = p.add_run(part)
            set_font(run, '宋体', 12, bold=True)
        else:
            run = p.add_run(part)
            set_font(run, '宋体', 12)
    return p

def add_table(doc, headers, rows):
    """添加表格"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    # 表头
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(header)
        set_font(run, '黑体', 11, bold=True)
        # 设置表头背景色
        shading = cell._element.get_or_add_tcPr()
        shd = shading.makeelement(qn('w:shd'), {
            qn('w:fill'): 'D9E2F3',
            qn('w:val'): 'clear'
        })
        shading.append(shd)
    # 数据行
    for row_idx, row in enumerate(rows):
        for col_idx, cell_text in enumerate(row):
            cell = table.rows[row_idx + 1].cells[col_idx]
            cell.text = ''
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            # 处理加粗
            parts = re.split(r'\*\*(.+?)\*\*', str(cell_text))
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    run = p.add_run(part)
                    set_font(run, '宋体', 10, bold=True)
                else:
                    run = p.add_run(part)
                    set_font(run, '宋体', 10)
    doc.add_paragraph()  # 表格后空行
    return table

def parse_markdown_table(lines, start_idx):
    """解析Markdown表格"""
    headers = [h.strip() for h in lines[start_idx].strip('|').split('|')]
    # 跳过分隔行
    rows = []
    idx = start_idx + 2
    while idx < len(lines) and lines[idx].strip().startswith('|'):
        row = [c.strip() for c in lines[idx].strip('|').split('|')]
        rows.append(row)
        idx += 1
    return headers, rows, idx

def convert_md_to_docx(md_path, docx_path):
    """将Markdown转换为docx"""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    doc = Document()

    # 设置默认字体
    style = doc.styles['Normal']
    style.font.name = '宋体'
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

    # 设置页边距
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.2)
        section.right_margin = Inches(1.2)

    idx = 0
    in_code_block = False
    code_lines = []

    while idx < len(lines):
        line = lines[idx]

        # 代码块
        if line.strip().startswith('```'):
            if in_code_block:
                # 结束代码块
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Pt(24)
                p.paragraph_format.line_spacing = 1.2
                for code_line in code_lines:
                    run = p.add_run(code_line + '\n')
                    set_font(run, 'Consolas', 10)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            idx += 1
            continue

        if in_code_block:
            code_lines.append(line)
            idx += 1
            continue

        # 空行
        if not line.strip():
            idx += 1
            continue

        # 标题
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()
            add_heading(doc, text, level)
            idx += 1
            continue

        # 表格
        if line.strip().startswith('|') and idx + 1 < len(lines) and '---' in lines[idx + 1]:
            headers, rows, new_idx = parse_markdown_table(lines, idx)
            add_table(doc, headers, rows)
            idx = new_idx
            continue

        # 列表
        if re.match(r'^\s*[-*]\s+', line):
            level = len(line) - len(line.lstrip())
            text = re.sub(r'^\s*[-*]\s+', '', line)
            add_list_item(doc, text, level // 2)
            idx += 1
            continue

        # 有序列表
        if re.match(r'^\s*\d+\.\s+', line):
            level = len(line) - len(line.lstrip())
            text = re.sub(r'^\s*\d+\.\s+', '', line)
            add_list_item(doc, text, level // 2)
            idx += 1
            continue

        # 水平线
        if line.strip() == '---':
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run('—' * 40)
            set_font(run, '宋体', 12, color=RGBColor(128, 128, 128))
            idx += 1
            continue

        # 普通段落
        add_paragraph(doc, line.strip())
        idx += 1

    doc.save(docx_path)
    print(f"文档已保存: {docx_path}")

if __name__ == '__main__':
    md_path = r'C:\Users\张三\Desktop\论文初稿\第二篇论文框架_AutoProCorrector.md'
    docx_path = r'C:\Users\张三\Desktop\论文初稿\第二篇论文框架_AutoProCorrector.docx'
    convert_md_to_docx(md_path, docx_path)
