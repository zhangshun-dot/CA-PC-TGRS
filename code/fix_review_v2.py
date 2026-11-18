# -*- coding: utf-8 -*-
r"""fix_review_v2.py — 修复一审意见中的其他问题
处理：
  1. §2.2标题精简
  2. §5.4观察3直觉补全
  3. §6.3 ρ口径速查
  4. 6类规模说明
  5. 创新点增量论证（引言加一段三者差异对照）
  6. 核心价值正面回应（引言强调物理先验+0.0461）
"""
from docx import Document

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v10.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v11.docx'

doc = Document(PAPER_PATH)
updates = []

def replace_in_para(idx, old, new, reason):
    p = doc.paragraphs[idx]
    if old in p.text:
        for run in p.runs:
            if old in run.text:
                run.text = run.text.replace(old, new)
                updates.append(f'[{idx}] {reason}')
                return True
        full = p.text.replace(old, new)
        p.runs[0].text = full
        for run in p.runs[1:]:
            run.text = ''
        updates.append(f'[{idx}] {reason} (跨run)')
        return True
    else:
        print(f'  警告: [{idx}] 未找到 "{old[:50]}"')
        return False

# ═══════════════════════════════════════════════════════════════
# 1. §2.2标题精简（段落17）
# ═══════════════════════════════════════════════════════════════
print('[1/6] §2.2标题精简...')
replace_in_para(17,
    '2.2 文本指定类别的跨域候选框重排序（开放词汇检测的子任务）与原型方法',
    '2.2 开放词汇检测与原型方法',
    '§2.2标题精简')

# ═══════════════════════════════════════════════════════════════
# 2. §5.4观察3直觉补全（段落91）
# ═══════════════════════════════════════════════════════════════
print('[2/6] §5.4观察3直觉补全...')
replace_in_para(91,
    '直觉：背景类跨域差异大时，正原型需要外推以适应目标域分布，而负原型外推会引入更多域偏移噪声，非对称设计（正原型外推、负原型保守）的收益更明显。',
    '直觉：背景类跨域差异大时，正原型需外推以适应目标域前景分布，而负原型外推会放大背景噪声——前者收益增大、后者损害增大，两者共同导致非对称设计的收益更明显。',
    '§5.4观察3直觉补全')

# ═══════════════════════════════════════════════════════════════
# 3. §6.3 ρ口径速查（段落108）
# ═══════════════════════════════════════════════════════════════
print('[3/6] §6.3 ρ口径速查...')
p108_text = doc.paragraphs[108].text
if '全数据Spearman ρ=+0.699' in p108_text:
    replace_in_para(108,
        '在37类/6域对上验证风险评分R(c)与A-CSPC增益的相关性：全数据Spearman ρ=+0.699',
        '在37类/6域对上验证风险评分R(c)与A-CSPC增益的相关性（口径速查：全数据ρ=+0.699 / 严格冻结留出ρ=+0.555 / LOSO阈值验证平均ρ=+0.686 / 6类主实验基准ρ=+0.354）：全数据Spearman ρ=+0.699',
        '§6.3 ρ口径速查')

# ═══════════════════════════════════════════════════════════════
# 4. 6类规模说明（段落40，§4.1）
# ═══════════════════════════════════════════════════════════════
print('[4/6] 6类规模说明...')
p40_text = doc.paragraphs[40].text
if '主实验选取源域DIOR侧3个映射源类' in p40_text:
    replace_in_para(40,
        '其中vehicle为粗粒度源域类别，映射到COCO的car/truck/bus/motorcycle四个细粒度类别，ship映射到boat。',
        '其中vehicle为粗粒度源域类别，映射到COCO的car/truck/bus/motorcycle四个细粒度类别，ship映射到boat。6类是遥感→自然场景共有类的硬约束（DIOR仅20类、COCO中可映射的共有类有限），而非偷懒选择；VOC→COCO 20类扩展实验（§4.6）验证了方法在更大类别规模上的普适性。',
        '§4.1 6类规模说明')

# ═══════════════════════════════════════════════════════════════
# 5. 创新点增量论证（引言段落12，贡献点之后加一段）
# ═══════════════════════════════════════════════════════════════
print('[5/6] 创新点增量论证...')
# 段落12是贡献点(3)，在其后插入一段三者差异对照
p12 = doc.paragraphs[12]._element
from docx.oxml.ns import qn
from lxml import etree

new_p = etree.SubElement(p12.getparent(), qn('w:p'))
p12.addnext(new_p)
run = etree.SubElement(new_p, qn('w:r'))
t = etree.SubElement(run, qn('w:t'))
t.text = '与已有原型方法的本质差异：CDVM（源域视觉原型+负原型抑制）仅在源域空间构造原型，未进行跨空间外推；TGE（文本引导外推）对正/负原型采用对称外推，未区分两者的不同迁移特性。本文的核心增量是"非对称构造"：正原型沿CLIP文本方向外推至目标域以适应成像域变化，负原型保留在源域空间以避免背景噪声放大——这一非对称设计在消融实验中验证了其必要性（对称外推EC配置增益从+0.0652降至+0.0034）。'
t.set(qn('xml:space'), 'preserve')
updates.append('[12后] 创新点增量论证：A-CSPC vs CDVM/TGE三者差异对照')

# ═══════════════════════════════════════════════════════════════
# 6. 核心价值正面回应（引言段落13，强调物理先验）
# ═══════════════════════════════════════════════════════════════
print('[6/6] 核心价值正面回应...')
p13_text = doc.paragraphs[13].text
if '本文的主张范围与限定' in p13_text:
    # 在段落13前插入一段强调物理先验的正面价值
    p13_elem = doc.paragraphs[13]._element
    new_p2 = etree.SubElement(p13_elem.getparent(), qn('w:p'))
    p13_elem.addprevious(new_p2)
    run2 = etree.SubElement(new_p2, qn('w:r'))
    t2 = etree.SubElement(run2, qn('w:t'))
    t2.text = '针对zero-shot场景下方法价值的质疑，本文进一步提出物理先验驱动的无目标域标注自适应参数控制：利用类别级物理先验（粒度失配、尺度、类内变异、类间混淆）预测逐类校正需求，直接驱动β_eff(c)=β_base·(1+α·R(c))的自适应超参数，在DIOR→COCO六类上取得+0.0461增益（固定参数仅+0.0166，相对提升177%），6/6类正增益，并在VOC→COCO 20类扩展中验证普适性。这一机制将"目标域调参"的主配置升级为"物理先验驱动的无标注自适应控制"，正面回应了zero-shot场景下的方法价值问题。'
    t2.set(qn('xml:space'), 'preserve')
    updates.append('[13前] 核心价值正面回应：物理先验驱动+0.0461')

# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════
doc.save(OUTPUT_PATH)
print(f'\n修复完成！共{len(updates)}处更新：')
for u in updates:
    print(f'  {u}')
print(f'\n保存至: {OUTPUT_PATH}')
