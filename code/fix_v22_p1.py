# -*- coding: utf-8 -*-
r"""fix_v22_p1.py — 修复第二优先级问题
1. §4.5标题与正文脱节
2. 表7/表8"四源域"但只有3行
3. β=0三值加口径速查
4. 反向迁移三值加口径速查
5. §6.5.3的604%加病态分母限定
"""
from docx import Document

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v21.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v22.docx'

doc = Document(PAPER_PATH)

def replace_paragraph_text(p, old_text, new_text):
    """替换段落中的文本"""
    if old_text in p.text:
        full_text = p.text.replace(old_text, new_text)
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = full_text
        else:
            p.add_run(full_text)
        return True
    return False

# ══════════════════════════ 修复1：§4.5标题 ══════════════════════════
print('修复1：§4.5标题与正文脱节')
for i, p in enumerate(doc.paragraphs):
    if '4.5' in p.text and '跨数据集' in p.text:
        old_text = p.text
        # 标题改为"反向迁移验证（COCO→DIOR与COCO→NWPU）"
        new_text = '4.5 反向迁移验证（COCO→DIOR与COCO→NWPU组件归因）'
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: {old_text[:60]} → {new_text}')
        break

# ══════════════════════════ 修复2：表7/表8"四源域" ══════════════════════════
print('\n修复2：表7/表8"四源域"但只有3行')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if '四源域' in text and ('表7' in text or '表8' in text or '跨源' in text):
        old_text = text
        # 改为"三源域（DIOR/NWPU/DOTA）+VisDrone正文单列"
        new_text = text.replace('四源域', '三源域')
        new_text = new_text.replace('DIOR/NWPU/DOTA/VisDrone', 'DIOR/NWPU/DOTA（VisDrone为4类不同粒度，单列于正文）')
        if old_text != new_text:
            for run in p.runs:
                run.text = ''
            if p.runs:
                p.runs[0].text = new_text
            else:
                p.add_run(new_text)
            print(f'  段落{i}: 四源域 → 三源域')
            print(f'    旧: {old_text[:80]}')
            print(f'    新: {new_text[:80]}')

# ══════════════════════════ 修复3：β=0三值加口径速查 ══════════════════════════
print('\n修复3：β=0三值加口径速查')
# 在§4.4附近添加口径速查
for i, p in enumerate(doc.paragraphs):
    if '4.4' in p.text and ('消融' in p.text or '组件' in p.text):
        print(f'  找到§4.4在段落{i}')
        # 在下一段添加口径速查
        break

# 搜索包含β=0三个值的段落
for i, p in enumerate(doc.paragraphs):
    if '−0.0005' in p.text or '-0.0005' in p.text:
        print(f'  段落{i}包含−0.0005: {p.text[:120]}')
        # 在该段落后添加口径速查
        old_text = p.text
        if '口径速查' not in old_text:
            new_text = old_text + '（β=0口径速查：全量六类−0.071 / N=100预过滤−0.0451；−0.0005为含负原型的另一消融配置，三者不可直接比较）'
            for run in p.runs:
                run.text = ''
            if p.runs:
                p.runs[0].text = new_text
            else:
                p.add_run(new_text)
            print(f'    已添加β=0口径速查')
        break

# ══════════════════════════ 修复4：反向迁移三值加口径速查 ══════════════════════════
print('\n修复4：反向迁移三值加口径速查')
for i, p in enumerate(doc.paragraphs):
    if '−0.0244' in p.text or '-0.0244' in p.text:
        print(f'  段落{i}包含−0.0244: {p.text[:120]}')
        old_text = p.text
        if '口径速查' not in old_text:
            new_text = old_text + '（反向迁移口径速查：−0.0244为COCO→DIOR 3类聚合N=20图 / −0.0866为六行细分协议 / −0.239为COCO→NWPU三类别控制变量实验，三者协议不同不可直接比较）'
            for run in p.runs:
                run.text = ''
            if p.runs:
                p.runs[0].text = new_text
            else:
                p.add_run(new_text)
            print(f'    已添加反向迁移口径速查')
        break

# ══════════════════════════ 修复5：§6.5.3的604%加病态分母限定 ══════════════════════════
print('\n修复5：§6.5.3的604%加病态分母限定')
for i, p in enumerate(doc.paragraphs):
    if '604%' in p.text or '604' in p.text and '相对提升' in p.text:
        print(f'  段落{i}包含604%: {p.text[:150]}')
        old_text = p.text
        if '病态分母' not in old_text and '固定配置' not in old_text:
            new_text = old_text + '该相对提升同样受固定配置选参质量影响（分母+0.0081来自开发域选参失效的固定配置），与§6.5.1的177%属同类病态分母问题，真实绝对增益为+0.0489。'
            for run in p.runs:
                run.text = ''
            if p.runs:
                p.runs[0].text = new_text
            else:
                p.add_run(new_text)
            print(f'    已添加604%病态分母限定')
        break

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('第二优先级修复完成')
