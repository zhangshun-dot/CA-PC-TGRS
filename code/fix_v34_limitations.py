# -*- coding: utf-8 -*-
r"""fix_v34_limitations.py — 精简局限性，扬长避短
"""
from docx import Document
from docx.oxml.ns import qn
from copy import deepcopy

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v33.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v34.docx'

doc = Document(PAPER_PATH)

def replace_paragraph_text(paragraph, new_text):
    """替换段落文本，保留第一个run的格式"""
    # 清除所有run
    for r in paragraph._p.findall(qn('w:r')):
        paragraph._p.remove(r)
    # 创建新run
    new_r = paragraph._p.makeelement(qn('w:r'), {})
    new_t = new_r.makeelement(qn('w:t'), {})
    new_t.text = new_text
    new_t.set(qn('xml:space'), 'preserve')
    new_r.append(new_t)
    paragraph._p.append(new_r)

print('=' * 60)
print('精简局限性部分')
print('=' * 60)

# 找到局限性段落（段落150）
p150 = doc.paragraphs[150]
old_text = p150.text
print(f'原文局限性: {len(old_text)}字')
print(f'原文: {old_text[:200]}...')

# 精简后的局限性（3点，每点一句话，用未来工作语气）
new_limitation = ('本文方法仍有可拓展的空间：'
    '(1)方法定位为弱校准开放词汇检测器的候选框重排序校正器，在强校准检测器（如OWLv2）上增益微弱或转负，'
    '未来可探索与检测器训练阶段联合优化以拓展适用范围；'
    '(2)物理先验预测当前基于37类/6域对的探索性验证，cluster bootstrap显示域对内相关较强，'
    '未来可在更多独立域对和自动化先验赋值上做更大规模确认；'
    '(3)主实验聚焦overhead→photo方向的跨域校正，反向方向（photo→overhead）因文本外推方向失配而收益有限，'
    '双向自适应外推是值得探索的方向。')

replace_paragraph_text(p150, new_limitation)
print(f'\n新文局限性: {len(new_limitation)}字')
print(f'新文: {new_limitation[:200]}...')
print(f'精简了 {len(old_text) - len(new_limitation)} 字')

print('\n' + '=' * 60)
print('检查结论部分，确保突出优势')
print('=' * 60)

# 找到结论段落（段落152）
p152 = doc.paragraphs[152]
old_conclusion = p152.text
print(f'原文结论: {len(old_conclusion)}字')
print(f'原文: {old_conclusion[:300]}...')

# 结论不需要大改，已经比较突出优势了
# 但可以检查是否有过多提及局限性的内容
print(f'\n结论中提及局限性的内容:')
if '局限' in old_conclusion:
    print('  包含"局限"字样')
if '失败' in old_conclusion:
    print('  包含"失败"字样')
if '转负' in old_conclusion:
    print('  包含"转负"字样')

print('\n结论部分已比较突出优势，不需要大改')

doc.save(OUTPUT_PATH)
print(f'\n\n论文已保存: {OUTPUT_PATH}')
print('局限性精简完成')
