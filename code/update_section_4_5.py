# -*- coding: utf-8 -*-
"""update_section_4_5.py — 把§4.5的改写整合到论文v34中，生成v35"""
from docx import Document
from docx.shared import Pt
from copy import deepcopy

# 读取v34
doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v34.docx')

# §4.5的段落索引
title_idx = 69  # 标题
body1_idx = 70  # 正文第一段（加粗）
body2_idx = 71  # 正文第二段（数据来源，不加粗）

# ── 1. 修改标题 ──
title_para = doc.paragraphs[title_idx]
# 清空原有runs
for run in title_para.runs:
    run.text = ''
# 设置新标题
title_para.runs[0].text = '4.5 反向迁移验证与粒度失配边界：质心聚集结构与方向选择性'

# ── 2. 准备改写后的正文内容 ──
new_paragraphs = [
    # 第一段：实验结果（保留原有数据，加粗）
    {
        'text': '反向 COCO→DIOR 迁移（3 个 DIOR 目标类聚合口径，N=20 图协议）用于验证 A-CSPC 外推组件的适用边界。【协议说明】§4.5 为跨数据集总体反向验证；§5.7 为固定 COCO→NWPU、三类别、同一 CLIP 编码下的组件控制变量实验。两者不进行数值直接比较。结果呈现与正向相反的规律：保守 B2（β=0，纯源域视觉原型）平均 +0.0418 且 3/3 类为正，说明自然场景源原型在遥感目标域上并未失效；完整外推模式平均 −0.0244（3/3 类为负），纯文本 T4 下降更多（−0.0793）。在 COCO→DIOR 六行细分协议下（§4.7），结果显示负原型在错误外推方向下有放大作用；在 COCO→NWPU 严格控制变量协议下（§5.7），进一步显示文本外推方向是主要损害来源，负原型在错误方向下进一步放大该损害。',
        'bold': True,
    },
    # 第二段：粒度失配与质心聚集结构（新增，不加粗）
    {
        'text': '粒度失配与质心聚集结构。反向迁移的失效并非单纯的"方向选择性"，而是具有更深层的几何根源。在 CLIP 联合特征空间中，细粒度类别原型（如 car、bus、truck、motorcycle）与它们的粗粒度聚合质心（vehicle）之间的余弦相似度均高于 0.95（car 0.979、bus 0.958、truck 0.990、motorcycle 0.965），表明细类原型已高度聚集在质心附近（夹角约 11°）。这一质心聚集结构意味着：在"细→粗"迁移中，源域细类原型与目标域粗类原型在特征空间中已高度对齐，任何形式的外推校正（无论是文本视角方向还是视觉质心收缩方向）都会引入额外噪声而无法产生收益——质心收缩方向实验中，向质心收缩的位移向量因 p_c ≈ c_coarse 而极小，其效果退化为近似不动，而"不动"（B2 保守）本身即取得 +0.0418 的最优增益。',
        'bold': False,
    },
    # 第三段：方向选择性的粒度解释（新增，不加粗）
    {
        'text': '方向选择性的粒度解释。上述发现将"方向选择性"提升为更本质的"粒度失配边界"：A-CSPC 的文本方向外推在校正"源域粗粒度→目标域细粒度"（质心→圆周，存在可弥合的语义间隙）时产生价值；而在"源域细粒度→目标域粗粒度"（圆周→质心，细类原型已聚集在质心附近）时，校正的前提（存在可弥合的间隙）不成立，外推反而引入噪声。这一边界由 CLIP 空间的几何结构决定，与具体外推方向的选择无关——多模板方向校准（减少 13.7% 损害）、目标域负原型替换（减少 27.3% 损害）、视觉质心收缩（损害降至 ≈0）均无法超越 B2 保守的 +0.0418，三者共同验证了"反向迁移不需要校正"这一几何结论。',
        'bold': False,
    },
    # 第四段：结论与定位（新增，不加粗）
    {
        'text': '因此，A-CSPC 应定位为：面向特定粒度方向（粗→细）的跨域原型校正器；在细→粗方向上，方法自动回退到 B2 保守模式（g₂ ≤ 0），源域视觉原型直接迁移即最优。方向门 g₂ 的域方向判别在 19 个域对上准确率 19/19，类级决策准确率 16/19（前者判断域方向，后者判断具体类别是否应外推）。（反向迁移口径速查：−0.0244 为 COCO→DIOR 3 类聚合 N=20 图 / −0.0866 为六行细分协议 / −0.239 为 COCO→NWPU 三类别控制变量实验，三者协议不同不可直接比较）',
        'bold': False,
    },
    # 第五段：数据来源说明（保留原有，不加粗）
    {
        'text': '本小节数据来源为 tge_probe_rev.json（N=20 图/类，DIOR 目标 3 类聚合，image-safe 逐图 AP50，COCO 源侧 500 图）；三个均值与逐类增益取平均一致（外推项从 JSON 4 位小数反算为 −0.0243，正文 −0.0244 与之一位末差，来自未舍入精度）。此口径与第 4.3 节表 4（N=100、6 行拆分）不同，图像池、类聚合粒度与基线均不可直接比较：vehicle 聚合行将车辆四子类的崩溃稀释为 −0.014，表 4 拆分后暴露 truck/bus/motorcycle 的 −0.14 至 −0.17。',
        'bold': False,
    },
]

# ── 3. 替换原有段落 ──
# 获取原有段落的样式参考
ref_body_para = doc.paragraphs[body1_idx]
ref_style = ref_body_para.style

# 清空原有两个正文段落
for idx in [body1_idx, body2_idx]:
    para = doc.paragraphs[idx]
    for run in para.runs:
        run.text = ''

# 在body1_idx位置写入第一段
para1 = doc.paragraphs[body1_idx]
run1 = para1.runs[0] if para1.runs else para1.add_run()
run1.text = new_paragraphs[0]['text']
run1.font.name = '宋体'
run1.font.size = Pt(11)
run1.font.bold = new_paragraphs[0]['bold']

# 在body1_idx后插入新段落
# 方法：在body2_idx位置写入第二段，然后在后面插入剩余段落
para2 = doc.paragraphs[body2_idx]
run2 = para2.runs[0] if para2.runs else para2.add_run()
run2.text = new_paragraphs[1]['text']
run2.font.name = '宋体'
run2.font.size = Pt(11)
run2.font.bold = new_paragraphs[1]['bold']

# 在body2_idx后插入剩余段落（第3、4、5段）
# 方法：在body2_idx段落后面插入新段落
insert_after_idx = body2_idx
for i in range(2, len(new_paragraphs)):
    # 在insert_after_idx后面插入新段落
    new_para = doc.paragraphs[insert_after_idx]._element.addnext(
        deepcopy(doc.paragraphs[insert_after_idx]._element)
    )
    # 获取新段落对象
    new_para_obj = doc.paragraphs[insert_after_idx + 1]
    # 清空内容
    for run in new_para_obj.runs:
        run.text = ''
    # 设置新内容
    run = new_para_obj.runs[0] if new_para_obj.runs else new_para_obj.add_run()
    run.text = new_paragraphs[i]['text']
    run.font.name = '宋体'
    run.font.size = Pt(11)
    run.font.bold = new_paragraphs[i]['bold']
    insert_after_idx += 1

# ── 4. 保存为v35 ──
output_path = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx'
doc.save(output_path)
print(f'已保存到: {output_path}')
print(f'§4.5已更新：标题加入"粒度失配边界：质心聚集结构与方向选择性"')
print(f'正文从2段扩展为5段，加入了质心聚集结构（cos>0.95）和粒度失配边界的机制解释')
