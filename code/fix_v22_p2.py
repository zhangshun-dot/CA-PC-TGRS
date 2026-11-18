# -*- coding: utf-8 -*-
r"""fix_v22_p2.py — 统一负号和引号
负号：- (U+002D) → − (U+2212)，但要排除URL、代码、邮箱等
引号：统一为中文引号""
"""
from docx import Document
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v22.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v22.docx'

doc = Document(PAPER_PATH)

def is_url_or_code(text):
    """判断是否是URL、代码、邮箱等不应替换负号的内容"""
    url_patterns = [
        r'https?://',
        r'www\.',
        r'\.com/',
        r'\.org/',
        r'\.edu/',
        r'\.gov/',
        r'github\.com',
        r'arxiv\.org',
        r'doi\.org',
    ]
    for pattern in url_patterns:
        if re.search(pattern, text):
            return True
    return False

def fix_minus_sign(text):
    """将负号从-替换为−，但排除URL和代码"""
    if is_url_or_code(text):
        return text
    # 替换数字前的负号：-数字 → −数字
    text = re.sub(r'(?<=\s)-(?=\d)', '−', text)
    text = re.sub(r'(?<=\()-(?=\d)', '−', text)
    # 替换+/-中的负号
    text = re.sub(r'(?<=\+)-(?=\d)', '−', text)
    # 替换"−0.xxx"格式（已经是正确的）
    return text

def fix_quotes(text):
    """统一引号为中文引号"""
    # 英文双引号 " " → 中文引号 ""
    # 这比较复杂，需要配对处理
    # 简单处理：将成对的"替换为""
    result = []
    in_quote = False
    for char in text:
        if char == '"':
            if not in_quote:
                result.append('"')
                in_quote = True
            else:
                result.append('"')
                in_quote = False
        else:
            result.append(char)
    return ''.join(result)

# 统计
minus_count = 0
quote_count = 0

# 处理所有段落
for p in doc.paragraphs:
    old_text = p.text
    new_text = fix_minus_sign(old_text)
    new_text = fix_quotes(new_text)
    if old_text != new_text:
        minus_count += old_text.count('-') - new_text.count('-')
        quote_count += old_text.count('"') - new_text.count('"')
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)

# 处理所有表格
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                old_text = p.text
                new_text = fix_minus_sign(old_text)
                new_text = fix_quotes(new_text)
                if old_text != new_text:
                    for run in p.runs:
                        run.text = ''
                    if p.runs:
                        p.runs[0].text = new_text
                    else:
                        p.add_run(new_text)

doc.save(OUTPUT_PATH)
print(f'负号替换: 约{minus_count}处')
print(f'引号替换: 约{quote_count}处')
print(f'论文已保存: {OUTPUT_PATH}')
