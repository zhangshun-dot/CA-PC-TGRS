# -*- coding: utf-8 -*-
"""
统计论文中的数学公式数量
"""
from docx import Document
import zipfile
import xml.etree.ElementTree as ET

doc_path = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v34.docx'

# 方法1：通过OMML XML统计公式
print("=" * 60)
print("方法1：通过OMML XML统计数学公式")
print("=" * 60)

with zipfile.ZipFile(doc_path, 'r') as z:
    with z.open('word/document.xml') as f:
        content = f.read().decode('utf-8')

# 统计oMath元素（独立公式）和oMathPara元素（公式段落）
omath_count = content.count('<m:oMath ')
omathpara_count = content.count('<m:oMathPara')
omath_self_close = content.count('<m:oMath/>')

print(f"独立公式 (m:oMath): {omath_count}")
print(f"公式段落 (m:oMathPara): {omathpara_count}")
print(f"自闭合公式: {omath_self_close}")
print(f"公式总数（去重估算）: {max(omath_count, omathpara_count)}")

# 方法2：通过python-docx遍历段落，查找包含公式的段落
print("\n" + "=" * 60)
print("方法2：遍历段落统计含公式的段落")
print("=" * 60)

doc = Document(doc_path)
formula_paragraphs = []
inline_formulas = 0

for i, para in enumerate(doc.paragraphs):
    # 检查段落XML中是否包含oMath
    para_xml = para._element.xml
    if 'm:oMath' in para_xml:
        # 统计该段落中的公式数量
        count = para_xml.count('<m:oMath ')
        formula_paragraphs.append((i+1, count, para.text[:50]))
        inline_formulas += count

print(f"含公式的段落数: {len(formula_paragraphs)}")
print(f"公式总数（按段落统计）: {inline_formulas}")

print("\n含公式的段落列表：")
for para_num, count, text in formula_paragraphs:
    print(f"  段落{para_num}: {count}个公式 - {text}...")

# 方法3：统计表格中的公式
print("\n" + "=" * 60)
print("方法3：统计表格中的公式")
print("=" * 60)

table_formulas = 0
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                if 'm:oMath' in para._element.xml:
                    table_formulas += para._element.xml.count('<m:oMath ')

print(f"表格中的公式数: {table_formulas}")

# 总结
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
total = inline_formulas + table_formulas
print(f"正文段落中的公式: {inline_formulas}")
print(f"表格中的公式: {table_formulas}")
print(f"公式总数: {total}")
