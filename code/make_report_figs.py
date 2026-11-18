# -*- coding: utf-8 -*-
"""make_report_figs.py — 技术报告1.docx 配图生成 (matplotlib, 白底学术风)
按占位描述生成矢量风格 PNG, 插入 docx 时再由 insert_figs.py 处理。
样式: 白底, 圆角框, 蓝/青/橙/红/灰配色, 中文用微软雅黑。
用法: python make_report_figs.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import matplotlib.image as mpimg

FONT = r'C:\Windows\Fonts\msyh.ttc'
fm.fontManager.addfont(FONT)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13

OUT = r'C:\Users\张三\Desktop\实验数据\_report_figs'
os.makedirs(OUT, exist_ok=True)

# 配色
NAVY='#1F3A6E'; BLUE='#2E6FD6'; TEAL='#13B5A6'; ORANGE='#E8833A'
RED='#E24A4A'; GREY='#8A93A6'; LGREY='#EAEef5'; DGREY='#5b6472'
GREEN='#3FB56B'; YELLOW='#E6B23A'; PURPLE='#8E6FD6'
WHITE='white'

def newfig(w=11.0, h=6.2, dpi=160):
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off')
    ax.set_facecolor('white'); fig.patch.set_facecolor('white')
    return fig, ax

def box(ax, x, y, w, h, text, fc=LGREY, ec=BLUE, tc='#1a1a1a',
        fs=12, bold=False, lw=1.8, rad=0.10, ha='center', va='center', wrap=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                      linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    if text:
        # 简单按换行符或手动\n处理
        ax.text(x+w/2, y+h/2, text, ha=ha, va=va, fontsize=fs,
                color=tc, fontweight='bold' if bold else 'normal', zorder=4,
                wrap=wrap, linespacing=1.25)
    return p

def arrow(ax, x1, y1, x2, y2, ec=GREY, lw=1.8, rad=0.12, style='->', off=0.0):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                          mutation_scale=14, linewidth=lw, color=ec,
                          connectionstyle=f"arc3,rad={rad}", zorder=2)
    ax.add_patch(arr)
    return arr

def txt(ax, x, y, s, fs=12, c='#1a1a1a', bold=False, ha='center', va='center', zorder=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c,
            fontweight='bold' if bold else 'normal', zorder=zorder, linespacing=1.2)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=fig.dpi, bbox_inches='tight',
                facecolor='white', pad_inches=0.12)
    plt.close(fig)
    print('saved', name)

# ───────────────────────── 图1: 跨域检测问题概念 ─────────────────────────
def fig1():
    fig, ax = newfig(11.6, 6.8)
    txt(ax, 5.8, 6.45, '图1  异构视觉平台下的跨域目标检测问题示意', fs=15, bold=True, c=NAVY)
    # 三列: (平台名, 主题色, 面板x0, 该平台的差异要点)
    cols = [
        ('遥感卫星', BLUE, 0.35, '观测视角: 俯视/大倾角\n目标尺度: 极小(像素级)\n背景: 地物/水体'),
        ('无人机',   TEAL, 4.05, '观测视角: 斜俯视\n目标尺度: 小—中\n背景: 街区/道路'),
        ('地面视觉', ORANGE, 7.75, '观测视角: 平视\n目标尺度: 大\n背景: 街景/室内'),
    ]
    pw, ph = 3.5, 3.0
    for name, col, x0, feat in cols:
        # 图像面板
        box(ax, x0, 3.1, pw, ph, '', fc='#f7f9fc', ec=col, lw=2.2, rad=0.06)
        cx = x0 + pw/2
        cy = 3.1 + ph/2 + 0.15
        if name == '遥感卫星':      # 俯视车顶
            ax.add_patch(Rectangle((cx-1.05, cy-0.45), 2.1, 0.9, fc=col, ec='none', alpha=.9, zorder=4))
            ax.add_patch(Rectangle((cx-0.45, cy-0.32), 0.9, 0.64, fc='white', ec=col, lw=1.2, zorder=5))
        elif name == '无人机':      # 斜俯视
            poly = Polygon([(cx-1.1, cy-0.25),(cx+0.9, cy-0.5),
                            (cx+1.1, cy+0.35),(cx-0.9, cy+0.5)],
                           closed=True, fc=col, ec='none', alpha=.9, zorder=4)
            ax.add_patch(poly)
            poly2 = Polygon([(cx-0.35, cy-0.18),(cx+0.35, cy-0.25),
                             (cx+0.4, cy+0.15),(cx-0.3, cy+0.2)],
                            closed=True, fc='white', ec=col, lw=1.2, zorder=5)
            ax.add_patch(poly2)
        else:                       # 平视侧影
            ax.add_patch(FancyBboxPatch((cx-1.1, cy-0.25), 2.2, 0.62,
                        boxstyle="round,pad=0,rounding_size=0.22",
                        fc=col, ec='none', alpha=.9, zorder=4))
            ax.add_patch(Rectangle((cx-0.55, cy+0.1), 1.0, 0.42, fc='white', ec=col, lw=1.2, zorder=5))
            for dx in (-0.65, 0.65):
                ax.add_patch(Circle((cx+dx, cy-0.25), 0.2, fc=DGREY, ec='none', zorder=5))
        txt(ax, cx, 2.8, name, fs=14, bold=True, c=col)
        # 差异要点
        txt(ax, cx, 2.15, feat, fs=11, c='#333')
        arrow(ax, cx, 3.1, cx, 2.5, ec=GREY, lw=1.2)
    # 底部结论
    box(ax, 0.35, 0.35, 10.9, 1.25,
        '同一类别进入不同视觉域后外观结构与特征分布发生变化 → 源域原型在目标域失效,\n'
        '基础检测器候选框排序失准, 跨域检测性能下降',
        fc='#eef4ff', ec=BLUE, tc=NAVY, fs=12.5, bold=True, rad=0.08)
    save(fig, 'fig1.png')

# ───────────────────────── 图7: A-CSPC 算法总体架构 ─────────────────────────
def fig7():
    fig, ax = newfig(12.4, 9.2)
    txt(ax, 6.2, 8.85, '图7  A-CSPC 算法总体架构图', fs=15, bold=True, c=NAVY)

    # 左侧: 源域知识输入
    box(ax, 0.3, 5.5, 2.5, 2.7, '', fc=NAVY, ec=NAVY, rad=0.08)
    txt(ax, 1.55, 7.95, '源域知识输入', fs=12.5, bold=True, c=WHITE)
    box(ax, 0.55, 7.15, 2.0, 0.55, '源域标注样本', fc='#dfe7f7', ec=BLUE)
    box(ax, 0.55, 6.4, 2.0, 0.55, '源域类别视觉原型', fc='#dfe7f7', ec=BLUE)
    box(ax, 0.55, 5.65, 2.0, 0.55, '负原型库', fc='#f7dede', ec=RED)

    # 上方: 文本语义支路
    box(ax, 3.3, 7.5, 3.0, 0.7, '文本语义支路', fc=PURPLE, ec=PURPLE, tc=WHITE, fs=12, bold=True, rad=0.08)
    box(ax, 3.3, 6.6, 1.4, 0.55, '源域文本模板', fc='#efe6fb', ec=PURPLE)
    box(ax, 4.9, 6.6, 1.4, 0.55, '目标域文本模板', fc='#efe6fb', ec=PURPLE)
    box(ax, 3.3, 5.85, 3.0, 0.55, '类别文本偏移方向', fc='#efe6fb', ec=PURPLE)
    box(ax, 6.8, 6.6, 1.7, 0.55, '通用名词集合', fc='#efe6fb', ec=PURPLE)
    box(ax, 6.8, 5.85, 1.7, 0.55, '共享域方向估计', fc='#e6f7f4', ec=TEAL)

    # 中部: 校正核心
    core = [
        ('共享域方向剥离', TEAL),
        ('正原型跨空间修正', BLUE),
        ('负原型源域保持', RED),
        ('方向门判断', ORANGE),
        ('类别级自适应参数生成', GREEN),
    ]
    cy0 = 4.85
    for i,(t,col) in enumerate(core):
        y = cy0 - i*0.82
        box(ax, 3.4, y, 4.9, 0.62, t, fc='#f3f6fc', ec=col, tc='#1a1a1a', fs=11.5, bold=True, rad=0.07)

    # 右侧: 目标图像支路
    box(ax, 8.7, 5.5, 3.4, 2.7, '', fc=NAVY, ec=NAVY, rad=0.08)
    txt(ax, 10.4, 7.95, '目标图像支路', fs=12.5, bold=True, c=WHITE)
    rchain = [
        '基础检测器 → 固定候选框',
        'CLIP 候选区域特征编码',
        '正负原型相似度计算',
        '视觉原型评分',
        '几何平均融合 (×原分数)',
        '候选框重新排序',
    ]
    for i,t in enumerate(rchain):
        y = 7.15 - i*0.52
        box(ax, 8.85, y, 3.1, 0.46, t, fc='#eef4ff', ec=BLUE, tc='#1a1a1a', fs=10, rad=0.05)

    # 连接箭头
    arrow(ax, 2.8, 6.7, 3.3, 7.0, ec=PURPLE)
    arrow(ax, 2.8, 5.9, 3.3, 6.2, ec=PURPLE)
    arrow(ax, 6.5, 6.1, 6.8, 6.4, ec=PURPLE)
    arrow(ax, 4.8, 5.85, 4.8, 5.47, ec=TEAL)
    arrow(ax, 8.3, 6.8, 8.7, 6.9, ec=BLUE)
    # 中部向下串联
    for i in range(len(core)-1):
        y1 = cy0 - i*0.82; y2 = cy0 - (i+1)*0.82
        arrow(ax, 5.85, y1, 5.85, y2+0.62, ec=GREY, lw=1.4)
    arrow(ax, 5.85, 4.85-4*0.82, 8.7, 6.3, ec=GREY, lw=1.4)

    # 底部约束标注
    box(ax, 0.3, 0.45, 11.8, 1.0,
        '边界与协议: 基础检测器参数冻结 · 候选框坐标与数量保持不变 · A-CSPC 新增可学习参数 = 0\n'
        '（仅做候选框分数融合与重排序, 不更新检测器权重）',
        fc='#fff5e6', ec=ORANGE, tc='#7a4a12', fs=11.5, bold=True, rad=0.08)
    save(fig, 'fig7.png')

# ───────────────────────── 图2: 传统适配 vs A-CSPC ─────────────────────────
def fig2():
    fig, ax = newfig(12.0, 5.8)
    txt(ax, 6.0, 5.45, '图2  传统跨域适配与 A-CSPC 迁移方式对比', fs=15, bold=True, c=NAVY)
    # 左: 传统
    box(ax, 0.4, 0.55, 5.4, 4.55, '', fc='#fff0f0', ec=RED, lw=2, rad=0.08)
    txt(ax, 3.1, 4.8, '传统跨域适配', fs=13, bold=True, c=RED)
    steps_l = ['① 目标域数据采集', '② 人工标注', '③ 模型重新训练', '④ 重新部署']
    for i,s in enumerate(steps_l):
        y = 3.85 - i*0.88
        box(ax, 0.75, y, 4.7, 0.62, s, fc='white', ec=RED, tc='#333', fs=12, rad=0.06)
        if i < 3:
            arrow(ax, 3.1, y, 3.1, y-0.26, ec=RED, lw=1.6)
    # 右: A-CSPC
    box(ax, 6.2, 0.55, 5.4, 4.55, '', fc='#eefaf3', ec=GREEN, lw=2, rad=0.08)
    txt(ax, 8.9, 4.8, 'A-CSPC (保留基础检测器)', fs=13, bold=True, c=GREEN)
    box(ax, 6.55, 2.4, 4.7, 1.7, '基础检测器 (参数冻结)\n↓\n候选框分数融合与重排序',
        fc='white', ec=GREEN, tc='#1a1a1a', fs=12.5, bold=True, rad=0.08)
    txt(ax, 8.9, 1.35, '无需目标域数据 · 无需重训练 · 即插即用', fs=11.5, c=GREEN, bold=True)
    save(fig, 'fig2.png')

# ───────────────────────── 图3: 覆盖不足 vs 排序失准 ─────────────────────────
def fig3():
    fig, ax = newfig(12.0, 5.8)
    txt(ax, 6.0, 5.45, '图3  候选框覆盖不足与候选框排序失校准对比', fs=15, bold=True, c=NAVY)
    # ── 左: 未覆盖 ──
    box(ax, 0.4, 0.55, 5.4, 4.55, '', fc='#f7f9fc', ec=GREY, lw=2, rad=0.06)
    txt(ax, 3.1, 4.8, '① 候选框覆盖不足', fs=13, bold=True, c=RED)
    # 图像区
    ax.add_patch(Rectangle((0.8, 1.15), 4.6, 3.1, fc='#e8edf5', ec='#c5cdd9', lw=1))
    # 真实目标 (星形位置)
    tgt = (4.45, 3.35)
    ax.add_patch(Circle(tgt, 0.34, fc=ORANGE, ec='white', lw=1.5, zorder=5))
    txt(ax, tgt[0], tgt[1], '目标', fs=10, c='white', ha='center', va='center', zorder=6, bold=True)
    # 三个候选框, 全都偏离目标
    for (bx,by,bw,bh) in [(1.05,1.5,0.95,0.75),(2.25,2.6,0.95,0.75),(1.6,3.3,0.95,0.75)]:
        ax.add_patch(Rectangle((bx,by), bw, bh, fill=False, ec=BLUE, lw=2.2, ls=(0,(4,3)), zorder=4))
    txt(ax, 2.6, 1.35, '候选框 (均未罩住真实目标)', fs=10.5, c=BLUE, ha='center')
    txt(ax, 3.1, 0.85, '真实目标未进入候选框集合 → A-CSPC 无法修复', fs=10.5, c=RED)
    # ── 右: 覆盖但排序低 ──
    box(ax, 6.2, 0.55, 5.4, 4.55, '', fc='#f7f9fc', ec=GREY, lw=2, rad=0.06)
    txt(ax, 8.9, 4.8, '② 候选框排序失校准', fs=13, bold=True, c=BLUE)
    ax.add_patch(Rectangle((6.6, 2.55), 4.6, 1.75, fc='#e8edf5', ec='#c5cdd9', lw=1))
    # 正确候选框罩住目标 (左上)
    ax.add_patch(Rectangle((6.95, 3.0), 1.1, 0.95, fill=True, fc=GREEN, ec=GREEN, alpha=.22, zorder=3))
    ax.add_patch(Rectangle((6.95, 3.0), 1.1, 0.95, fill=False, ec=GREEN, lw=2.4, zorder=4))
    tgt2 = (7.5, 3.47)
    ax.add_patch(Circle(tgt2, 0.3, fc=ORANGE, ec='white', lw=1.5, zorder=5))
    txt(ax, tgt2[0], tgt2[1], '目标', fs=9.5, c='white', ha='center', va='center', zorder=6, bold=True)
    # 错误高分框 (右上)
    ax.add_patch(Rectangle((9.6, 3.0), 1.2, 0.95, fill=False, ec=RED, lw=2.2, zorder=4))
    txt(ax, 10.2, 3.47, '背景', fs=9.5, c=RED, ha='center', va='center', zorder=5)
    # 排序分数条 (下方)
    bar_x = 6.95
    txt(ax, bar_x, 2.32, '候选框排序:', fs=10, c='#333', ha='left', bold=True)
    ax.add_patch(Rectangle((bar_x, 1.85), 2.2, 0.3, fc=RED, ec='none'))
    txt(ax, bar_x+2.35, 2.0, 'rank1  0.41 (背景框)', fs=10, c=RED, ha='left', va='center')
    ax.add_patch(Rectangle((bar_x, 1.35), 0.62, 0.3, fc=GREEN, ec='none'))
    txt(ax, bar_x+2.35, 1.5, 'rank5  0.12 (目标框)', fs=10, c=GREEN, ha='left', va='center')
    txt(ax, 8.9, 0.85, 'A-CSPC 主要作用于此类: 校正分数与排序', fs=10.5, c=BLUE, bold=True)
    save(fig, 'fig3.png')

# ───────────────────────── 图4: 粗粒度 vehicle 原型 ─────────────────────────
def fig4():
    import numpy as np
    fig, ax = newfig(11.0, 6.0)
    txt(ax, 5.5, 5.65, '图4  源域粗粒度 vehicle 原型与细粒度类别特征关系', fs=15, bold=True, c=NAVY)
    ax.set_xlim(0,11); ax.set_ylim(0,6); ax.axis('off')
    rng = np.random.default_rng(7)
    # 粗粒度 vehicle 中心
    vc = (5.5, 3.0)
    ax.add_patch(Circle(vc, 0.45, fc=NAVY, ec='none', alpha=.9, zorder=3))
    txt(ax, vc[0], vc[1], 'vehicle', fs=11, c='white', ha='center', va='center', zorder=5, bold=True)
    # 细粒度类别
    fines = [('car',BLUE,(3.2,4.0)),('bus',TEAL,(4.0,1.9)),
             ('truck',ORANGE,(7.0,4.1)),('motorcycle',PURPLE,(7.4,1.8))]
    for name,col,(x,y) in fines:
        pts = rng.normal((x,y),0.28,size=(18,2))
        ax.scatter(pts[:,0],pts[:,1], s=26, c=col, alpha=.55, edgecolors='none', zorder=2)
        ax.add_patch(Circle((x,y),0.32, fill=False, ec=col, lw=2, zorder=3))
        arrow(ax, vc[0]+np.sign(x-vc[0])*0.5, vc[1]+np.sign(y-vc[1])*0.5,
              x-np.sign(x-vc[0])*0.4, y-np.sign(y-vc[1])*0.4, ec=GREY, lw=1.4)
        txt(ax, x, y+0.55, name, fs=11, c=col, bold=True, ha='center')
    txt(ax, 5.5, 0.5, '粗粒度类别中心与各细粒度类别之间存在明显粒度间隙 → 直接迁移排序失准',
        fs=11.5, c=NAVY, bold=True)
    save(fig, 'fig4.png')

# ───────────────────────── 图6: 总体技术路线图 ─────────────────────────
def fig6():
    fig, ax = newfig(12.8, 7.6)
    txt(ax, 6.4, 7.25, '图6  A-CSPC 总体技术路线图', fs=15, bold=True, c=NAVY)
    layers = [
        ('① 跨域问题分析', BLUE, 0),
        ('② 实验可修复性诊断', TEAL, 1),
        ('③ A-CSPC 校正方法', ORANGE, 2),
        ('④ 边界控制', RED, 3),
        ('⑤ 实验与工程验证', GREEN, 4),
    ]
    ly0 = 6.35
    yl = {}
    for name,col,idx in layers:
        y = ly0 - idx*1.28
        yl[idx] = y
        box(ax, 0.4, y, 3.1, 0.95, name, fc=col, ec=col, tc='white', fs=12.5, bold=True, rad=0.08)
        if idx < 4:
            arrow(ax, 1.95, y, 1.95, y-0.33, ec=GREY, lw=1.6)
    # 右侧三个内容框 (标题+说明都在框内)
    # 诊断支路 ↔ 层②
    box(ax, 3.9, 4.9, 8.5, 1.35, '', fc='#e6f7f4', ec=TEAL, lw=2, rad=0.07)
    txt(ax, 8.15, 5.9, '实验诊断支路 (需目标域GT, 仅用于实验分析)', fs=12.5, bold=True, c='#0e7a6c')
    txt(ax, 8.15, 5.3, '候选框覆盖率 · 排序AUC · 排序改善量 → 可修复性分析', fs=11, c='#1a6b5a')
    arrow(ax, 3.5, yl[1]+0.5, 3.9, 5.6, ec=TEAL, lw=1.8)
    # 推理主路 ↔ 层③④
    box(ax, 3.9, 2.8, 8.5, 1.65, '', fc='#fff5e6', ec=ORANGE, lw=2, rad=0.07)
    txt(ax, 8.15, 4.1, '实际推理主路 (不使用目标域GT)', fs=12.5, bold=True, c='#7a4a12')
    txt(ax, 8.15, 3.5, '源域视觉原型 → 非对称正负原型校正 → 共享域方向剥离 →\n方向门判断 → 类别级自适应参数 → 分数融合与重排序',
        fs=10.5, c='#7a4a12')
    arrow(ax, 3.5, yl[2]+0.5, 3.9, 3.9, ec=ORANGE, lw=1.8)
    arrow(ax, 3.5, yl[3]+0.5, 3.9, 3.35, ec=ORANGE, lw=1.8)
    # 验证 ↔ 层⑤
    box(ax, 3.9, 0.75, 8.5, 1.3, '', fc='#eefaf3', ec=GREEN, lw=2, rad=0.07)
    txt(ax, 8.15, 1.7, '实验与工程验证', fs=12.5, bold=True, c=GREEN)
    txt(ax, 8.15, 1.15, '核心性能验证 · 跨源域验证 · 跨检测器验证 · 失败边界分析 · 工程原型 · 低空异构视觉应用验证',
        fs=10.5, c='#1e7a4a')
    arrow(ax, 3.5, yl[4]+0.5, 3.9, 1.4, ec=GREEN, lw=1.8)
    save(fig, 'fig6.png')

# ───────────────────────── 图8: 分层可修复性诊断 ─────────────────────────
def fig8():
    fig, ax = newfig(11.0, 6.4)
    txt(ax, 5.5, 6.05, '图8  A-CSPC 分层可修复性诊断', fs=15, bold=True, c=NAVY)
    ax.set_xlim(0,11); ax.set_ylim(0,6.2); ax.axis('off')
    # 底板
    ax.add_patch(Rectangle((0.8,0.8),9.4,4.8, fc='#f7f9fc', ec=GREY, lw=1.5))
    # 左半: 低覆盖 (不适用)
    ax.add_patch(Rectangle((0.8,0.8),4.7,4.8, fc='#fff5e6', ec='none', alpha=.75))
    # 右上: 高覆盖·高排序
    ax.add_patch(Rectangle((5.5,3.2),4.7,2.4, fc='#eefaf3', ec='none', alpha=.8))
    # 右下: 高覆盖·低排序 (A-CSPC 目标)
    ax.add_patch(Rectangle((5.5,0.8),4.7,2.4, fc='#eaf1fd', ec='none', alpha=.8))
    # 分界虚线
    ax.plot([5.5,5.5],[0.8,5.6], ls=(0,(4,3)), c=GREY, lw=1.2)
    ax.plot([5.5,10.2],[3.2,3.2], ls=(0,(4,3)), c=GREY, lw=1.2)
    txt(ax, 3.15, 3.2, '低覆盖\n(A-CSPC 不适用:\n候选框缺失, 无法修复)', fs=11.5, c='#b06a12', bold=True)
    txt(ax, 7.85, 4.4, '高覆盖 · 高排序\n(无需校正)', fs=11.5, c=GREEN, bold=True)
    txt(ax, 7.85, 2.0, '高覆盖 · 低排序\n(A-CSPC 主要作用)', fs=12, c=BLUE, bold=True)
    # 散点
    ax.scatter([6.3,7.2,8.0,9.2],[4.2,4.7,4.0,4.5], s=60, c=GREEN, zorder=4)
    ax.scatter([1.8,2.6,3.6,4.4],[1.6,2.2,1.4,2.6], s=55, c=ORANGE, zorder=4)
    ax.scatter([6.4,7.6,8.8,9.5],[2.5,1.6,2.6,1.9], s=65, c=BLUE, zorder=4)
    # 轴
    txt(ax, 5.5, 0.45, '候选框覆盖能力 →', fs=11, c='#333')
    ax.text(0.45, 3.2, '原型排序质量 →', fontsize=11, rotation=90, va='center', ha='right', c='#333')
    save(fig, 'fig8.png')

# ───────────────────────── 图9: 非对称原型校正 ─────────────────────────
def fig9():
    import numpy as np
    fig, ax = newfig(11.6, 6.6)
    txt(ax, 5.8, 6.25, '图9  A-CSPC 非对称原型校正示意', fs=15, bold=True, c=NAVY)
    ax.set_xlim(0,11.6); ax.set_ylim(0,6.4); ax.axis('off')
    # 目标域正原型区域 (右侧大圈)
    tp=(8.1,2.6)
    ax.add_patch(Circle(tp,1.2, fc=GREEN, ec=GREEN, lw=1.2, alpha=.15))
    txt(ax, 9.5, 1.05, '目标域正原型区域', fs=11.5, c=GREEN, bold=True, ha='center')
    # 源域正原型 → 校正后正原型
    sp=(2.0,1.9)
    ax.add_patch(Circle(sp,0.42, fc=BLUE, ec='none', alpha=.9, zorder=4))
    txt(ax, sp[0], sp[1]-0.75, '源域正原型', fs=11, c=BLUE, bold=True, ha='center')
    newp=(5.9,2.7)
    ax.add_patch(Circle(newp,0.42, fc=TEAL, ec='none', alpha=.9, zorder=4))
    arrow(ax, sp[0]+0.45, sp[1]+0.2, newp[0]-0.45, newp[1]-0.1, ec=TEAL, lw=2.2)
    txt(ax, 3.6, 2.9, '沿文本语义方向外推', fs=10.5, c=TEAL, ha='center')
    txt(ax, 7.0, 2.15, '校正后正原型\n(进入目标域区域)', fs=10.5, c='#0e8a7a', bold=True, ha='left')
    # 负原型: 保持在源域
    np_=(2.9,5.0)
    ax.add_patch(Circle(np_,0.4, fc=RED, ec='none', alpha=.9, zorder=4))
    txt(ax, np_[0], np_[1]+0.62, '负原型 (源域保持)', fs=11, c=RED, bold=True, ha='center')
    # 对称外推反例: 推入目标域
    sym=(7.5,3.35)
    ax.add_patch(Circle(sym,0.4, fc=RED, ec=RED, lw=1.5, alpha=.35, zorder=3))
    arrow(ax, np_[0]+0.42, np_[1]-0.2, sym[0]-0.35, sym[1]+0.3, ec=RED, lw=1.8, style='->')
    txt(ax, 9.9, 4.85, '若对称外推: 负原型\n进入目标域 → 产生干扰', fs=10, c=RED, ha='center')
    # 底部结论
    txt(ax, 5.8, 0.4, '非对称设计: 正原型外推、负原型保持 → 负原型不污染目标域有效特征区域',
        fs=11.5, c=NAVY, bold=True)
    save(fig, 'fig9.png')

# ───────────────────────── 图10: 文本方向分解 ─────────────────────────
def fig10():
    import numpy as np
    fig, ax = newfig(11.6, 6.2)
    txt(ax, 5.8, 5.85, '图10  文本外推方向中类别语义成分与共享域成分分解', fs=14.5, bold=True, c=NAVY)
    ax.set_xlim(0,11.6); ax.set_ylim(0,6); ax.axis('off')
    O=(1.6,1.6)
    ax.add_patch(Circle(O,0.11, fc=DGREY, ec='none', zorder=5))
    txt(ax, O[0]-0.15, O[1]-0.3, '源域', fs=10.5, c=DGREY, ha='center')
    # 共享域方向 (粗灰, 向上)
    S=(2.9,4.6)
    arrow(ax, O[0], O[1], S[0], S[1], ec=GREY, lw=3.2)
    txt(ax, S[0]+0.15, S[1]+0.25, '共享域方向 (各类近似一致)', fs=11, c=DGREY, bold=True, ha='left')
    # car: 原始偏移 = 共享域 + 判别分量 (平行四边形)
    A=(6.0,4.3)
    arrow(ax, O[0], O[1], A[0], A[1], ec=BLUE, lw=2.6)
    ax.plot([S[0],A[0]],[S[1],A[1]], ls=(0,(4,3)), c=BLUE, lw=1.6)
    txt(ax, A[0]+0.25, A[1]+0.1, 'car 原始偏移', fs=10.5, c=BLUE, bold=True, ha='left')
    mid=((S[0]+A[0])/2, (S[1]+A[1])/2)
    txt(ax, mid[0]+0.35, mid[1], '判别方向分量\n(类别特有)', fs=10, c=BLUE, ha='left')
    # 其他类别: 都" leaning"向共享方向
    for name,col,tip in [('bus',TEAL,(7.2,3.3)),('truck',ORANGE,(4.4,4.9)),('motorcycle',PURPLE,(7.9,4.7))]:
        arrow(ax, O[0], O[1], tip[0], tip[1], ec=col, lw=2.0)
        txt(ax, tip[0]+0.12, tip[1]+0.18, name, fs=10, c=col, bold=True, ha='left')
    # 右侧结论框
    box(ax, 8.6, 0.6, 2.85, 1.9, '', fc='#eef4ff', ec=BLUE, lw=1.6, rad=0.08)
    txt(ax, 10.0, 1.55, '多个类别的原始\n文本偏移均包含\n近似一致的共享域\n方向, 同时保留各\n类别判别分量',
        fs=10, c=NAVY)
    save(fig, 'fig10.png')

# ───────────────────────── 图11: 边界感知与自适应控制 ─────────────────────────
def fig11():
    fig, ax = newfig(12.6, 7.0)
    txt(ax, 6.3, 6.65, '图11  A-CSPC 边界感知与类别级自适应控制流程', fs=15, bold=True, c=NAVY)
    # 输入
    box(ax, 4.2, 5.6, 3.6, 0.7, '文本域方向 d + 类别粒度', fc='#eef4ff', ec=BLUE, fs=12, bold=True, rad=0.08)
    arrow(ax, 6.0, 5.6, 6.0, 5.2, ec=GREY, lw=1.8)
    # 方向门
    box(ax, 4.4, 4.35, 3.2, 0.85, '方向门判断', fc=ORANGE, ec=ORANGE, tc='white', fs=13, bold=True, rad=0.18)
    # 不满足 → 保守回退 (左)
    arrow(ax, 4.4, 4.77, 3.35, 4.77, ec=RED, lw=2.0)
    txt(ax, 3.87, 5.0, '不满足', fs=10.5, c=RED, bold=True)
    box(ax, 0.5, 3.6, 2.85, 2.2, '', fc='#fdeaea', ec=RED, lw=2, rad=0.08)
    txt(ax, 1.92, 5.5, '保守回退分支', fs=12, bold=True, c=RED)
    txt(ax, 1.92, 4.7, '关闭正原型外推\n仅保留负原型抑制\n\n候选框采用\n检测器原始分数',
        fs=10.5, c='#8a1f1f')
    # 满足 → 风险评分 (右)
    arrow(ax, 7.6, 4.77, 8.05, 4.77, ec=GREEN, lw=2.0)
    txt(ax, 7.82, 5.0, '满足', fs=10.5, c=GREEN, bold=True)
    box(ax, 8.05, 3.6, 4.15, 2.25, '', fc='#eefaf3', ec=GREEN, lw=2, rad=0.08)
    txt(ax, 10.12, 5.55, '失配风险评分', fs=12, bold=True, c=GREEN)
    txt(ax, 10.12, 4.6, '粒度失配 · 尺度敏感性\n类内变异 · 类间混淆', fs=11, c='#1e5c38')
    arrow(ax, 10.12, 3.6, 10.12, 3.15, ec=GREEN, lw=1.8)
    box(ax, 7.75, 2.35, 4.45, 0.8, '类别级自适应参数 (α, λ)', fc='#eefaf3', ec=GREEN, fs=12, bold=True, rad=0.08)
    arrow(ax, 9.97, 2.35, 9.97, 1.9, ec=GREEN, lw=1.8)
    box(ax, 5.9, 1.05, 6.3, 0.85, '自适应原型校正 → 候选框分数融合与重排序',
        fc=NAVY, ec=NAVY, tc='white', fs=12, bold=True, rad=0.08)
    arrow(ax, 1.92, 3.6, 1.92, 1.47, ec=RED, lw=1.8)
    arrow(ax, 1.92, 1.47, 5.9, 1.47, ec=RED, lw=1.8)
    txt(ax, 2.6, 1.72, '保守模式', fs=10, c=RED)
    save(fig, 'fig11.png')

# ───────────────────────── 图12: 信息使用协议与推理流程 ─────────────────────────
def fig12():
    fig, ax = newfig(13.0, 7.2)
    txt(ax, 6.5, 6.85, '图12  A-CSPC 信息使用协议与完整推理流程', fs=15, bold=True, c=NAVY)
    # 左: 实验诊断 (灰虚线)
    box(ax, 0.35, 1.0, 3.9, 5.3, '', fc='#f4f5f7', ec=GREY, lw=1.8, rad=0.08)
    ax.add_patch(Rectangle((0.35,1.0), 3.9, 5.3, fill=False, ec=GREY, lw=1.4, ls=(0,(5,4))))
    txt(ax, 2.3, 6.0, '实验诊断区域', fs=13, bold=True, c=DGREY)
    txt(ax, 2.3, 5.55, '(使用目标域真实框 GT)', fs=10.5, c=DGREY)
    for i,t in enumerate(['候选框覆盖率','源域原型排序AUC','外推前后排序改善量']):
        box(ax, 0.6, 4.75-i*0.72, 3.4, 0.55, t, fc='white', ec=GREY, tc='#333', fs=10.5, rad=0.05)
    txt(ax, 2.3, 2.5, '→ 失效机理分析\n→ 可修复性研究\n→ 方法边界验证', fs=10.5, c=DGREY)
    box(ax, 0.55, 1.2, 3.5, 0.85, '仅实验阶段使用\n不进入实际无标注推理', fc='white', ec=GREY,
        tc=DGREY, fs=10, bold=True, rad=0.06)
    # 中: 离线准备
    box(ax, 4.55, 1.0, 3.9, 5.3, '', fc='#fff5e6', ec=ORANGE, lw=2, rad=0.08)
    txt(ax, 6.5, 6.0, '推理前离线准备', fs=13, bold=True, c='#7a4a12')
    txt(ax, 6.5, 5.55, '(类别级信息, 可预先计算缓存)', fs=10, c='#7a4a12')
    prep = ['源域正/负视觉原型构建','源域与目标域文本嵌入','共享域方向估计与剥离','类别映射',
            '方向门计算','类别失配风险评分']
    for i,t in enumerate(prep):
        box(ax, 4.8, 4.8-i*0.62, 3.4, 0.5, t, fc='white', ec=ORANGE, tc='#5a3a10', fs=10, rad=0.05)
    # 右: 实际推理
    box(ax, 8.75, 1.0, 3.9, 5.3, '', fc='#eef4ff', ec=BLUE, lw=2, rad=0.08)
    txt(ax, 10.7, 6.0, '目标图像实际推理', fs=13, bold=True, c=BLUE)
    txt(ax, 10.7, 5.55, '(逐图在线执行)', fs=10.5, c=BLUE)
    inf = ['基础检测器 → 固定候选框','候选区域CLIP特征提取','正负原型匹配','类别级自适应评分',
           '检测器分数×原型分数融合','候选框重排序输出']
    for i,t in enumerate(inf):
        box(ax, 9.0, 4.75-i*0.56, 3.4, 0.48, t, fc='white', ec=BLUE, tc='#1a3a6e', fs=10, rad=0.05)
    box(ax, 9.0, 1.14, 3.4, 0.7, '不使用目标域真实标注\n不在线训练 · 不更新检测器参数',
        fc='white', ec=BLUE, tc='#1a3a6e', fs=9.5, bold=True, rad=0.06)
    # 区域间箭头
    arrow(ax, 4.25, 3.6, 4.55, 3.6, ec=GREY, lw=1.8)
    arrow(ax, 8.45, 3.6, 8.75, 3.6, ec=GREY, lw=1.8)
    save(fig, 'fig12.png')

# ───────────────────────── 图18: 方向门 24 场景判定 (示意) ─────────────────────────
def fig18():
    fig, ax = newfig(11.6, 6.4)
    txt(ax, 5.8, 6.05, '图18  方向门在 24 个跨域场景上的判定结果 (示意)', fs=14.5, bold=True, c=NAVY)
    dirs = ['DIOR→COCO','DIOR→VisDrone','COCO→DIOR','VisDrone→DIOR']
    cats = ['car','bus','truck','motorcycle','boat','airplane']
    # 示意门控: 与 g2/τ* 比较结果一致的排版 (1=外推, 0=回退)
    gate = [
        [1,1,0,0,1,1],
        [1,0,0,0,1,0],
        [0,1,1,1,0,1],
        [0,1,0,1,1,0],
    ]
    x0, y0, cw, ch = 2.1, 1.35, 1.45, 0.95
    for j,c in enumerate(cats):
        txt(ax, x0+j*cw+cw/2, y0+4*ch+0.3, c, fs=11, c=NAVY, bold=True)
    for i,d in enumerate(dirs):
        txt(ax, x0-0.25, y0+(3-i)*ch+ch/2, d, fs=10.5, c=NAVY, bold=True, ha='right')
        for j in range(6):
            v = gate[i][j]
            col = GREEN if v else '#f0b429'
            name = '外推' if v else '回退'
            ax.add_patch(Rectangle((x0+j*cw+0.06, y0+(3-i)*ch+0.06), cw-0.12, ch-0.12,
                         fc=col, ec='white', lw=1.5, alpha=.85))
            txt(ax, x0+j*cw+cw/2, y0+(3-i)*ch+ch/2, name, fs=10.5,
                c='white' if v else '#5a4300', bold=True)
    # 图例
    ax.add_patch(Rectangle((2.1, 0.55), 0.5, 0.35, fc=GREEN, alpha=.85))
    txt(ax, 2.75, 0.72, '方向门开启 → 正原型外推', fs=10.5, c='#1e5c38', ha='left')
    ax.add_patch(Rectangle((5.6, 0.55), 0.5, 0.35, fc='#f0b429', alpha=.85))
    txt(ax, 6.25, 0.72, '方向门关闭 → 保守回退', fs=10.5, c='#7a5c00', ha='left')
    txt(ax, 5.8, 0.15, '示意布局: 具体开/关以真实实验的门控输出为准', fs=9.5, c=GREY)
    save(fig, 'fig18.png')

# ───────────────────────── 图19: 增益 vs 检测器饱和程度 (真实数据) ─────────────────────────
def fig19():
    import json, numpy as np
    fig, ax = newfig(11.4, 6.4)
    txt(ax, 5.7, 6.15, '图19  A-CSPC 逐类增益 vs 基础检测器性能 (AP50)', fs=14.5, bold=True, c=NAVY)
    ax.set_xlim(0,11.4); ax.set_ylim(0,6.1); ax.axis('off')
    files = [('gdino-tiny', BLUE, 'C:/Users/张三/Desktop/实验数据/aic_comp_gdino_tiny.json'),
             ('gdino-base', TEAL, 'C:/Users/张三/Desktop/实验数据/aic_comp_gdino_base.json'),
             ('owlv2',      ORANGE,'C:/Users/张三/Desktop/实验数据/aic_comp_owlv2.json')]
    px0, py0, pw, ph = 1.8, 0.95, 8.6, 4.2
    ax.add_patch(Rectangle((px0,py0), pw, ph, fc='#f9fafc', ec=GREY, lw=1.2))
    GMIN, GMAX = -0.08, 0.22
    def ymap(g): return py0 + (g-GMIN)/(GMAX-GMIN)*ph
    # 增益=0 线
    ax.plot([px0,px0+pw],[ymap(0),ymap(0)], c=GREY, lw=1.2, ls=(0,(4,3)))
    txt(ax, px0+pw-0.1, ymap(0)+0.16, '增益 = 0', fs=9.5, c=GREY, ha='right')
    for name, col, fp in files:
        d = json.load(open(fp, encoding='utf-8'))
        for cls, v in d['per_class'].items():
            b = v['baseline']['ap50']; a = v['A-CSPC']['ap50']
            x = px0 + b/0.65*pw
            y = ymap(a-b)
            ax.scatter([x],[y], s=75, c=col, zorder=4, edgecolors='white', linewidths=1.2)
            txt(ax, x+0.12, y+0.15, cls, fs=8.5, c=col, ha='left', bold=True)
    txt(ax, px0+pw/2, py0-0.35, 'Baseline AP50 (检测器饱和程度) →', fs=11, c='#333')
    ax.text(px0-0.28, py0+ph/2, 'A-CSPC 增益 (ΔAP50) →', fontsize=11, rotation=90,
            va='center', ha='right', c='#333')
    # 图例
    for i,(name,col,_) in enumerate(files):
        ax.scatter([px0+0.35+i*2.7],[py0+ph+0.32], s=85, c=col, edgecolors='white')
        txt(ax, px0+0.65+i*2.7, py0+ph+0.32, name, fs=11, c='#333', ha='left')
    txt(ax, 5.7, 0.25, '数据来源: aic_comp_gdino_tiny / gdino_base / owlv2 实验输出 (6 类逐类 AP50)',
        fs=10, c=GREY)
    save(fig, 'fig19.png')

# ───────────────────────── 图22: 低空异构视觉应用映射 ─────────────────────────
def fig22():
    fig, ax = newfig(12.6, 7.0)
    txt(ax, 6.3, 6.65, '图22  低空异构视觉应用映射与验证场景', fs=15, bold=True, c=NAVY)
    # 左: 平台与数据
    plats = [('遥感平台', BLUE, 'DIOR · NWPU VHR-10 · DOTA', 4.9),
             ('无人机平台', TEAL, 'VisDrone', 3.1),
             ('地面视觉平台', ORANGE, 'COCO · VOC', 1.3)]
    for name, col, ds, y in plats:
        box(ax, 0.4, y, 3.1, 1.35, '', fc='#f7f9fc', ec=col, lw=2, rad=0.08)
        txt(ax, 1.95, y+1.0, name, fs=12.5, bold=True, c=col)
        txt(ax, 1.95, y+0.4, ds, fs=10, c='#333')
        arrow(ax, 3.5, y+0.67, 4.5, 3.35, ec=col, lw=1.8)
    # 中: A-CSPC 模块
    box(ax, 4.5, 2.4, 3.4, 1.9, '', fc=NAVY, ec=NAVY, rad=0.1)
    txt(ax, 6.2, 3.95, 'A-CSPC 校正模块', fs=13, bold=True, c='white')
    txt(ax, 6.2, 3.0, '基础检测器之后的\n独立候选框校正\n(文本语义方向+边界控制)', fs=10, c='#cdd9f0')
    arrow(ax, 7.9, 3.35, 8.9, 3.35, ec=GREY, lw=2)
    # 右: 应用方向
    apps = ['低空巡检','交通目标监测','机场周边感知','水域目标识别']
    for i,a in enumerate(apps):
        y = 4.7 - i*0.95
        box(ax, 8.9, y, 3.3, 0.7, a, fc='#eefaf3', ec=GREEN, tc='#1e5c38', fs=11.5, rad=0.08)
    # 标注: 已验证 vs 潜在
    box(ax, 0.4, 0.25, 11.8, 0.7,
        '公开数据实验验证 (已开展): DIOR / NWPU / DOTA / VisDrone / COCO / VOC    |    潜在工程应用 (未部署): 右列应用方向',
        fc='#fffde6', ec='#b09a2a', tc='#6e5c12', fs=10.5, bold=True, rad=0.08)
    save(fig, 'fig22.png')

# ───────────────────────── 图20: 典型候选框校正案例 (示意图) ─────────────────────────
def fig20():
    import numpy as np
    fig, ax = newfig(12.4, 7.0)
    txt(ax, 6.2, 6.6, '图20  A-CSPC 典型候选框校正案例对比 (示意图)', fs=14.5, bold=True, c=NAVY)
    # 示意图 banner
    ax.add_patch(Rectangle((8.55, 6.0), 3.75, 0.5, fill=False, ec=RED, lw=1.6, ls=(0,(5,4))))
    txt(ax, 10.42, 6.25, '示意图 · 投稿前替换为真实样本', fs=9.5, c=RED, bold=True)
    samples = ['_samples/uns_0.jpg', '_samples/uns_1.jpg', '_samples/uns_2.jpg']
    panels = [
        ('① 正常修复', GREEN,  '目标框原本排序靠后 →\nA-CSPC 提升其分数并前移', 0.35),
        ('② 稳定保持', BLUE,  '目标框原本已正确 →\nA-CSPC 维持原排序', 4.45),
        ('③ 保守回退', ORANGE, '目标模糊/失配 →\n方向门关闭, 沿用原分数', 8.55),
    ]
    pw = 3.5
    for k, (title, col, desc, x0) in enumerate(panels):
        box(ax, x0, 1.3, pw, 4.5, '', fc='none', ec=col, lw=2, rad=0.08)
        txt(ax, x0+pw/2, 5.45, title, fs=13, bold=True, c=col)
        # 图像区
        ix, iy, iw, ih = x0+0.3, 2.5, pw-0.6, 2.2
        img = None
        try:
            img = mpimg.imread(samples[k])
            H, W = img.shape[0], img.shape[1]
            a, ba = W/H, iw/ih
            if a > ba:
                dw, dh = iw, iw/a; x0b, y0b = ix, iy+(ih-dh)/2
            else:
                dh, dw = ih, ih*a; y0b, x0b = iy, ix+(iw-dw)/2
            ax.imshow(img, extent=(x0b, x0b+dw, y0b, y0b+dh), zorder=1, aspect='auto')
        except Exception as e:
            ax.add_patch(Rectangle((ix,iy), iw, ih, fc='#eef1f6', ec='#c5cdd9', lw=1.2, zorder=1))
        # 图像边框
        ax.add_patch(Rectangle((ix,iy), iw, ih, fill=False, ec='#9aa3b2', lw=1.2, zorder=2))
        # 车辆中心(示意): 取图像区中部偏下
        cx, cy = ix+iw/2, iy+ih*0.55
        bw, bh = iw*0.44, ih*0.52
        if title.startswith('①'):
            ax.add_patch(Rectangle((ix+iw*0.06, iy+ih*0.08), iw*0.30, ih*0.26,
                                    fill=False, ec=RED, lw=2.4, zorder=4))
            txt(ax, ix+iw*0.21, iy+ih*0.08-0.16, '原排序低 (rank5)', fs=8.5, c=RED, bold=True)
            ax.add_patch(Rectangle((cx-bw/2, cy-bh/2), bw, bh, fill=False, ec=GREEN, lw=2.8, zorder=5))
            txt(ax, cx, cy-bh/2-0.16, 'A-CSPC 校正后 (rank1)', fs=8.5, c=GREEN, bold=True)
        elif title.startswith('②'):
            ax.add_patch(Rectangle((cx-bw/2, cy-bh/2), bw, bh, fill=False, ec=GREEN, lw=2.8, zorder=5))
            txt(ax, cx, cy-bh/2-0.16, '已正确 → 维持 (rank1)', fs=8.5, c=GREEN, bold=True)
        else:
            ax.add_patch(Rectangle((cx-bw/2, cy-bh/2), bw, bh, fill=False, ec=ORANGE, lw=2.4,
                                   ls=(0,(4,3)), zorder=5))
            txt(ax, cx, cy-bh/2-0.16, '门关闭 → 沿用原分数', fs=8.5, c=ORANGE, bold=True)
        txt(ax, ix+0.06, iy+0.06, '示意底图(CC0)', fs=7.5, c='#888', ha='left', va='bottom', zorder=6)
        txt(ax, x0+pw/2, 1.78, desc, fs=9.5, c='#444')
    txt(ax, 6.2, 0.5, '示意图: 以真实照片作底图演示三种校正模式, 框位为示意; 真实案例请改用你实验的检测样本',
        fs=10, c=GREY, bold=True)
    save(fig, 'fig20.png')

# ───────────────────────── 图21: 工程验证原型界面 (示意图) ─────────────────────────
def fig21():
    fig, ax = newfig(12.6, 7.2)
    txt(ax, 6.3, 6.85, '图21  A-CSPC 工程验证原型界面 (示意图)', fs=14.5, bold=True, c=NAVY)
    ax.add_patch(Rectangle((9.0, 6.2), 3.5, 0.5, fill=False, ec=RED, lw=1.6, ls=(0,(5,4))))
    txt(ax, 10.75, 6.45, '示意图 · 投稿前换真实截图', fs=9.5, c=RED, bold=True)
    # 左: 输入
    box(ax, 0.35, 1.2, 3.6, 5.2, '', fc='#f7f9fc', ec=BLUE, lw=2, rad=0.08)
    txt(ax, 2.15, 6.05, '输入区', fs=12.5, bold=True, c=BLUE)
    box(ax, 0.6, 4.5, 3.1, 1.25, '图像输入\n[ 拖入 / 选择图片 ]', fc='white', ec=BLUE, tc='#333', fs=11, rad=0.06)
    box(ax, 0.6, 3.15, 3.1, 1.05, '目标类别输入\ncar; bus; truck; motorcycle', fc='white', ec=BLUE, tc='#333', fs=10.5, rad=0.06)
    box(ax, 0.6, 1.55, 3.1, 1.25, '基础检测器选择\n[ Grounding DINO-tiny ]', fc='white', ec=BLUE, tc='#333', fs=10.5, rad=0.06)
    # 中: 输出对比
    box(ax, 4.2, 1.2, 4.4, 5.2, '', fc='#fbfcfe', ec=GREY, lw=1.8, rad=0.08)
    txt(ax, 6.4, 6.05, '推理输出对比', fs=12.5, bold=True, c=DGREY)
    box(ax, 4.45, 4.5, 3.9, 1.2, 'Baseline 输出\n检出: 2/3 目标\n排序: 目标框 rank5 (低)',
        fc='#fdeaea', ec=RED, tc='#8a1f1f', fs=10.5, rad=0.06)
    box(ax, 4.45, 3.0, 3.9, 1.2, 'A-CSPC 输出\n检出: 3/3 目标\n排序: 目标框 rank1 (前移)',
        fc='#eefaf3', ec=GREEN, tc='#1e5c38', fs=10.5, rad=0.06)
    arrow(ax, 6.4, 4.5, 6.4, 4.2, ec=GREY, lw=1.6)
    txt(ax, 8.85, 3.6, '↑ 校正', fs=10, c=GREEN, bold=True, ha='left')
    # 右: 状态
    box(ax, 8.8, 1.2, 3.4, 5.2, '', fc='#fff5e6', ec=ORANGE, lw=2, rad=0.08)
    txt(ax, 10.5, 6.05, '状态 / 协议', fs=12.5, bold=True, c='#7a4a12')
    box(ax, 9.05, 4.6, 2.9, 1.0, 'Safe-Correction 门\n状态: 开启 / 关闭',
        fc='white', ec=ORANGE, tc='#5a3a10', fs=10.5, rad=0.06)
    box(ax, 9.05, 3.2, 2.9, 1.1, '融合分数\nfused = √(gd × vis)\nvis = max(0, pos − λ·neg)',
        fc='white', ec=ORANGE, tc='#5a3a10', fs=9.5, rad=0.06)
    box(ax, 9.05, 1.55, 2.9, 1.3, '边界\n无目标域标注 · 不在线训练\n参数冻结 · 即插即用',
        fc='white', ec=ORANGE, tc='#5a3a10', fs=9.5, bold=True, rad=0.06)
    txt(ax, 6.3, 0.5, '示意图: 按报告占位符描述的原型界面布局绘制, 实际界面请截图替换', fs=10, c=GREY, bold=True)
    save(fig, 'fig21.png')

if __name__ == '__main__':
    import numpy as np
    for f in [fig1,fig2,fig3,fig4,fig6,fig7,fig8,fig9,fig10,fig11,fig12,fig18,fig19,fig20,fig21,fig22]:
        f()
    print('all figs done ->', OUT)
