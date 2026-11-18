# -*- coding: utf-8 -*-
from docx import Document
doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v7.docx')

print('=== 论文基本信息 ===')
print(f'总段落数: {len(doc.paragraphs)}')
print(f'总表格数: {len(doc.tables)}')
print()

print('=== P0修改验证 ===')
checks = [
    (40, '源域DIOR侧3个映射源类', 'P0-1 §4.1类别列表'),
    (52, 'N=100预过滤口径', 'P0-2 覆盖率口径'),
    (122, '自动化赋值的物理先验', 'P0-3 α扫描先验来源'),
    (134, '更大规模验证', 'P0-4 结论自动化'),
]
for idx, keyword, name in checks:
    text = doc.paragraphs[idx].text
    if keyword in text:
        print(f'  OK {name}')
    else:
        print(f'  FAIL {name}: 未找到"{keyword}"')
print()

print('=== P1修改验证 ===')
checks = [
    (43, '−0.0005', 'P1-1 纯视觉原型数值'),
    (43, '表2', 'P1-3 表引用'),
    (43, '全量六类口径', 'P1-4 CDVM全量'),
    (56, 'N=100口径', 'P1-4 CDVM N=100'),
    (81, '§4.5', 'P1-6 表15归属'),
    (108, '仅6类主实验口径', 'P1-7 ρ基准'),
    (122, '补充材料表S4', 'P1-8 boat逐类'),
]
for idx, keyword, name in checks:
    text = doc.paragraphs[idx].text
    if keyword in text:
        print(f'  OK {name}')
    else:
        print(f'  FAIL {name}: 未找到"{keyword}"')
print()

print('=== P2修改验证 ===')
checks = [
    (91, '差异较大时最优', 'P2-2 观察3表述'),
    (3, '人工先验对照+0.0478', 'P2-4 headline数字'),
    (26, '面向跨域目标检测的非对称原型校正', 'P2-5 图1标题'),
]
for idx, keyword, name in checks:
    text = doc.paragraphs[idx].text
    if keyword in text:
        print(f'  OK {name}')
    else:
        print(f'  FAIL {name}: 未找到"{keyword}"')
print()

print('=== 主结果数值确认 ===')
p43 = doc.paragraphs[43].text
if '+0.0646' in p43:
    print('  OK 主结果+0.0646（原方法几何平均）')
else:
    print('  FAIL 主结果数值异常')
print()
print('全部验证完成！')
