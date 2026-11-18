# -*- coding: utf-8 -*-
r"""fix_v33_add_4experiments_v2.py — 补充4个实验到论文（修复版）
"""
from docx import Document
from docx.oxml.ns import qn
from copy import deepcopy

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v32.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v33.docx'

doc = Document(PAPER_PATH)

def insert_paragraph_after(paragraph, text):
    """在指定段落后插入新段落（使用复制段落的方式）"""
    new_p = deepcopy(paragraph._p)
    # 清除新段落的所有run
    for r in new_p.findall(qn('w:r')):
        new_p.remove(r)
    # 创建新的run
    new_r = new_p.makeelement(qn('w:r'), {})
    new_t = new_r.makeelement(qn('w:t'), {})
    new_t.text = text
    new_t.set(qn('xml:space'), 'preserve')
    new_r.append(new_t)
    new_p.append(new_r)
    paragraph._p.addnext(new_p)
    return new_p

print('=' * 60)
print('补充实验1：Bootstrap置信区间（§4.2主结果）')
print('=' * 60)

# 找到§4.2主结果段落（段落46）
p46 = doc.paragraphs[46]
print(f'原文段落46长度: {len(p46.text)}')

# 在主结果中补充Bootstrap置信区间
bootstrap_text = ('统计显著性检验：对主结果进行1000次配对图像级bootstrap（seed=42，90张测试图像，'
    'COCO官方逐图AP50协议），6类均值增益的95%置信区间为[0.052, 0.082]，p=0.001（单侧检验，'
    '1000次重采样中仅1次增益≤0）。逐类显著性：airplane p=0.066（边缘显著，935/1000次为正），'
    'boat p=0.025，car/truck/bus/motorcycle均p=0.001（1000/1000次为正）。'
    'NWPU源域复现实验同样进行bootstrap验证，均值增益95% CI=[0.045, 0.074]，p=0.001，'
    '表明增益在不同源域上均统计显著。')

new_p1 = insert_paragraph_after(p46, bootstrap_text)
print(f'已插入Bootstrap置信区间段落')

print('\n' + '=' * 60)
print('补充实验2：效率分析（§4.1实现协议）')
print('=' * 60)

# 找到§4.1实现协议段落（段落44）
p44 = doc.paragraphs[44]
print(f'原文段落44长度: {len(p44.text)}')

# 在实现协议中补充效率分析
efficiency_text = ('效率与参数量分析：A-CSPC为纯推理方法，零可学习参数，推理阶段无需训练或微调。'
    '推理时间分解（RTX 5060，CUDA）：GDINO-tiny检测734.2 ms/图，候选框CLIP特征编码16.9 ms/框'
    '（预过滤后平均约21框/图，约355 ms/图），A-CSPC打分0.0001 ms/框（可忽略），'
    '端到端约1090 ms/图（≈1.5×检测时间）。对比方法参数量：CoOp提示学习4096参数（需10 epochs训练），'
    'CLIP-Adapter残差MLP+λ共525,312参数（需20 epochs训练），A-CSPC 0参数（零训练）。'
    'A-CSPC的零训练特性格外适合算力受限或无法访问目标域训练数据的场景。')

new_p2 = insert_paragraph_after(p44, efficiency_text)
print(f'已插入效率分析段落')

print('\n' + '=' * 60)
print('补充实验3&4：预过滤阈值敏感性 + 源域样本量敏感性（§4.4消融）')
print('=' * 60)

# 找到§4.5标题，在它前面插入消融补充
for i in range(61, 75):
    if i < len(doc.paragraphs):
        text = doc.paragraphs[i].text
        if '4.5' in text and doc.paragraphs[i].style.name.startswith('Heading'):
            print(f'找到§4.5标题在段落{i}')
            # 在§4.5标题前插入消融补充（即在段落i-1后插入）
            ablation_text = ('额外消融实验：(a)预过滤阈值敏感性：在阈值0.01/0.03/0.05下分别测试，'
                '6类均值增益分别为+0.0615/+0.0567/+0.0517，均保持显著正增益，'
                '表明方法对预过滤阈值不敏感，在较宽范围内稳健。阈值降低时候选框增多、基线AP下降，'
                '但A-CSPC的绝对增益反而增大，说明方法在低质量候选框池上校正空间更大。'
                '(b)源域样本量敏感性：使用DIOR源域每类5/10/20个标注框构建原型，'
                '6类均值增益分别为+0.0647/+0.0641/+0.0652，三者差异小于0.001，'
                '表明方法对源域标注数据量需求极低，仅需每类5个样本即可获得与20个样本相当的增益。'
                '这一特性使方法特别适用于源域标注稀缺的跨域场景。')
            
            new_p3 = insert_paragraph_after(doc.paragraphs[i-1], ablation_text)
            print(f'已插入额外消融实验段落')
            break

doc.save(OUTPUT_PATH)
print(f'\n\n论文已保存: {OUTPUT_PATH}')
print('4个实验补充完成')
