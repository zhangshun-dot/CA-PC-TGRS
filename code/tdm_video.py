# -*- coding: utf-8 -*-
"""tdm_video.py — TDM (Token Distribution Modulation) 原理+效果 视频
黑底荧光板书风, 同源 A-CSPC 视频风格。10 幕 / 共 155s / 1280x720 / 25fps。
帧序列输出后由 ffmpeg 合成, 再铺 _tpl_audio.m4a (同前一条抖音音轨)。

运行: C:/Users/张三/.workbuddy/binaries/python/envs/default/Scripts/python.exe tdm_video.py
"""
import os, math
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageFont

W, H = 1280, 720
FPS = 25
OUT = r'C:\Users\张三\Desktop\实验数据\_tdm_frames'
os.makedirs(OUT, exist_ok=True)

FONT_B = r'C:\Windows\Fonts\msyhbd.ttc'
FONT_R = r'C:\Windows\Fonts\msyh.ttc'
_fonts = {}
def font(sz, bold=True):
    k = (sz, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT_R, sz)
    return _fonts[k]

# 荧光配色 (同 A-CSPC)
GREEN  = (57, 255, 136)
BLUE   = (77, 166, 255)
YELLOW = (255, 214, 61)
PURPLE = (199, 125, 255)
RED    = (255, 77, 109)
WHITE  = (245, 245, 245)
GREY   = (140, 140, 155)
BG     = (10, 10, 14)

def dim(c, a):
    a = max(0.0, min(1.0, a))
    return (int(c[0]*a), int(c[1]*a), int(c[2]*a))

def ease(x):
    x = max(0.0, min(1.0, x))
    return x*x*(3-2*x)

def seg(p, a, b):
    return ease((p - a) / (b - a)) if b > a else 1.0

# ── 荧光绘制原语 (gd=glow层, sd=清晰层) ──
def L(gd, sd, p1, p2, color, a=1.0, w=4):
    c = dim(color, a)
    gd.line([p1, p2], fill=c, width=w+5)
    sd.line([p1, p2], fill=c, width=w)

def dash(gd, sd, p1, p2, color, a=1.0, w=4, dlen=14, gap=10):
    x1, y1 = p1; x2, y2 = p2
    d = math.hypot(x2-x1, y2-y1)
    if d < 1: return
    n = int(d // (dlen+gap)) + 1
    for i in range(n):
        s = i*(dlen+gap)/d; e = min((i*(dlen+gap)+dlen)/d, 1.0)
        if s >= 1: break
        L(gd, sd, (x1+(x2-x1)*s, y1+(y2-y1)*s), (x1+(x2-x1)*e, y1+(y2-y1)*e), color, a, w)

def arrow(gd, sd, p1, p2, color, a=1.0, w=4):
    x1, y1 = p1; x2, y2 = p2
    ang = math.atan2(y2-y1, x2-x1)
    hl = 18
    bx, by = x2 - hl*math.cos(ang), y2 - hl*math.sin(ang)
    L(gd, sd, p1, (bx, by), color, a, w)
    la = (x2 - hl*math.cos(ang-0.45), y2 - hl*math.sin(ang-0.45))
    ra = (x2 - hl*math.cos(ang+0.45), y2 - hl*math.sin(ang+0.45))
    c = dim(color, a)
    gd.polygon([(x2, y2), la, ra], fill=c)
    sd.polygon([(x2, y2), la, ra], fill=c)

def dot(gd, sd, p, color, a=1.0, r=10):
    x, y = p
    gd.ellipse([x-r-4, y-r-4, x+r+4, y+r+4], fill=dim(color, a))
    sd.ellipse([x-r, y-r, x+r, y+r], fill=dim(color, a))

def ring(gd, sd, p, r, color, a=1.0, w=4):
    x, y = p
    gd.ellipse([x-r, y-r, x+r, y+r], outline=dim(color, a), width=w+5)
    sd.ellipse([x-r, y-r, x+r, y+r], outline=dim(color, a), width=w)

def rect(gd, sd, x0, y0, x1, y1, color, a=1.0, w=4):
    L(gd, sd, (x0, y0), (x1, y0), color, a, w)
    L(gd, sd, (x1, y0), (x1, y1), color, a, w)
    L(gd, sd, (x1, y1), (x0, y1), color, a, w)
    L(gd, sd, (x0, y1), (x0, y0), color, a, w)

def frect(gd, sd, x0, y0, x1, y1, color, a=1.0):
    c = dim(color, a)
    gd.rectangle([x0, y0, x1, y1], fill=c)
    sd.rectangle([x0, y0, x1, y1], fill=c)

def dashed_ring(gd, sd, center, rx, ry, color, a=1.0, w=3):
    steps = 72
    for i in range(0, steps, 2):
        a0 = i*2*math.pi/steps
        a1 = (i+1)*2*math.pi/steps
        p1 = (center[0]+rx*math.cos(a0), center[1]+ry*math.sin(a0))
        p2 = (center[0]+rx*math.cos(a1), center[1]+ry*math.sin(a1))
        L(gd, sd, p1, p2, color, a, w)

def txt(gd, sd, p, s, size, color, a=1.0, bold=True, anchor='mm'):
    f = font(size, bold)
    c = dim(color, a)
    gd.text(p, s, font=f, fill=c, anchor=anchor)
    sd.text(p, s, font=f, fill=c, anchor=anchor)

def satellite(gd, sd, c, color, a):
    x, y = c
    rect(gd, sd, x-26, y-18, x+26, y+18, color, a, 4)
    L(gd, sd, (x-26, y), (x-66, y), color, a, 3)
    L(gd, sd, (x+26, y), (x+66, y), color, a, 3)
    for sx in (-1, 1):
        px = x + sx*92
        rect(gd, sd, px-26, y-16, px+26, y+16, color, a, 3)
    L(gd, sd, (x, y+18), (x, y+48), color, a, 3)
    dot(gd, sd, (x, y+56), color, a, 5)

# ── 十幕 ──
def scene1(gd, sd, p):
    a1 = seg(p, 0.00, 0.22); a2 = seg(p, 0.18, 0.45); a3 = seg(p, 0.42, 0.70)
    satellite(gd, sd, (250, 250), BLUE, a1)
    txt(gd, sd, (250, 362), 'DIOR 遥感源域', 30, WHITE, a1)
    txt(gd, sd, (250, 402), '小目标 · 低信噪比', 24, GREY, a1)
    bx0, by, bw, bh = 620, 230, 560, 46
    txt(gd, sd, (bx0, by-26), '裸标签', 24, WHITE, a2)
    frect(gd, sd, bx0, by, bx0+int(bw*0.2125/0.33), by+bh, GREY, a2*0.6)
    rect(gd, sd, bx0, by, bx0+int(bw*0.2125/0.33), by+bh, GREY, a2, 3)
    txt(gd, sd, (bx0+int(bw*0.2125/0.33)+12, by+bh/2), 'DR 0.2125', 22, WHITE, a2)
    by2 = by+92
    txt(gd, sd, (bx0, by2-26), '+ 前缀', 24, GREEN, a2)
    w2 = int(bw*0.3025/0.33)
    frect(gd, sd, bx0, by2, bx0+w2, by2+bh, GREEN, a2*0.7)
    rect(gd, sd, bx0, by2, bx0+w2, by2+bh, GREEN, a2, 3)
    txt(gd, sd, (bx0+w2+12, by2+bh/2), 'DR 0.3025', 22, GREEN, a2)
    arrow(gd, sd, (bx0+int(bw*0.2125/0.33), by+bh/2), (bx0+w2, by2+bh/2), GREEN, a2, 4)
    by3 = by2+92
    txt(gd, sd, (bx0, by3-26), '堆修饰词', 24, RED, a3)
    # 论文只给"反而下降", 不杜撰精确 DR; 条形视觉上短于前缀即可
    w3 = int(bw*0.250/0.33)
    frect(gd, sd, bx0, by3, bx0+w3, by3+bh, RED, a3*0.45)
    rect(gd, sd, bx0, by3, bx0+w3, by3+bh, RED, a3, 3)
    txt(gd, sd, (bx0+w3+12, by3+bh/2), '↓ 反而下降', 22, RED, a3)
    arrow(gd, sd, (bx0+w2, by2+bh/2), (bx0+w3, by3+bh/2), RED, a3, 4)
    txt(gd, sd, (640, 600), 'less-is-more 异常: 词越多 ≠ 越好', 28, YELLOW, a3)
    return seg(p, 0.66, 0.88)

def scene2(gd, sd, p):
    a1 = seg(p, 0.00, 0.25); a2 = seg(p, 0.22, 0.50); a3 = seg(p, 0.50, 0.78)
    rect(gd, sd, 140, 200, 560, 470, RED, a1, 4)
    txt(gd, sd, (350, 235), '静态 [CLS] 池化', 28, RED, a1)
    txt(gd, sd, (350, 300), '所有 token 压扁为 1 向量', 22, WHITE, a1)
    txt(gd, sd, (350, 350), 'DPI: 必然丢失', 24, YELLOW, a1)
    txt(gd, sd, (350, 390), '文本–图像互信息', 24, YELLOW, a1)
    rect(gd, sd, 720, 200, 1140, 470, BLUE, a2, 4)
    txt(gd, sd, (930, 235), '跨注意力 动态加权', 28, BLUE, a2)
    txt(gd, sd, (930, 300), '按 query 逐 token 聚合', 22, WHITE, a2)
    txt(gd, sd, (930, 350), '保留 token 级结构', 24, GREEN, a2)
    txt(gd, sd, (930, 390), '互信息更完整', 24, GREEN, a2)
    arrow(gd, sd, (560, 335), (720, 335), YELLOW, a3, 5)
    txt(gd, sd, (640, 545), 'DPI 信息瓶颈: 静态池化 = 信息流失', 28, YELLOW, a3)
    return seg(p, 0.74, 0.92)

def scene3(gd, sd, p):
    a1 = seg(p, 0.00, 0.30); a2 = seg(p, 0.30, 0.60); a3 = seg(p, 0.60, 0.85)
    ox, oy, ex, ey = 200, 520, 1080, 180
    L(gd, sd, (ox, oy), (ex, oy), GREY, a1, 3)
    L(gd, sd, (ox, oy), (ox, ey), GREY, a1, 3)
    txt(gd, sd, (ex, oy+30), 'token 数', 22, GREY, a1)
    txt(gd, sd, (ox-30, ey-10), '互信息', 22, GREY, a1)
    dash(gd, sd, (240, 360), (1040, 250), GREEN, a2, 4)
    txt(gd, sd, (560, 300), '供给: 语义上限 (衰减)', 22, GREEN, a2)
    dash(gd, sd, (240, 500), (1040, 330), RED, a2, 4)
    txt(gd, sd, (820, 480), '需求: 噪声累积', 22, RED, a2)
    dot(gd, sd, (640, 305), YELLOW, a3, 12)
    txt(gd, sd, (640, 248), 'n_c ≈ 7–12', 26, YELLOW, a3)
    txt(gd, sd, (640, 585), '越过阈值 → 噪声主导, token 越多越糟', 26, YELLOW, a3)
    return seg(p, 0.80, 0.92)

def scene4(gd, sd, p):
    a1 = seg(p, 0.00, 0.30); a2 = seg(p, 0.30, 0.60); a3 = seg(p, 0.60, 0.85)
    txt(gd, sd, (640, 250), '条件语义容量上界', 30, WHITE, a1)
    txt(gd, sd, (640, 332), 'C = 7.9 bit', 48, YELLOW, a1)
    txt(gd, sd, (640, 402), '× 注意力衰减 × 跨模型保持', 26, GREY, a2)
    bx0, by, bw, bh = 340, 470, 600, 50
    frect(gd, sd, bx0, by, bx0+int(bw*0.5), by+bh, PURPLE, a3*0.7)
    rect(gd, sd, bx0, by, bx0+int(bw*0.5), by+bh, PURPLE, a3, 3)
    txt(gd, sd, (bx0+bw/2, by+bh/2), 'TDM 达 42–58% 上界', 24, WHITE, a3)
    return seg(p, 0.80, 0.92)

def scene5(gd, sd, p):
    a1 = seg(p, 0.00, 0.30); a2 = seg(p, 0.30, 0.62); a3 = seg(p, 0.60, 0.85)
    txt(gd, sd, (640, 220), 'OFAT 七轴语义测量', 30, WHITE, a1)
    axes = ['语法前缀', '粒度', '空间', '几何', '尺度', '光谱', '纹理']
    for i, ax in enumerate(axes):
        col = i // 4; row = i % 4
        x = 430 + col*440; y = 320 + row*70
        ring(gd, sd, (x-100, y), 16, GREEN, a2, 3)
        txt(gd, sd, (x, y), ax, 26, WHITE, a2, anchor='lm')
    dashed_ring(gd, sd, (980, 560), 120, 78, BLUE, a3, 3)
    txt(gd, sd, (980, 560), 'PCA', 24, BLUE, a3)
    txt(gd, sd, (980, 650), '7轴解释 84.2% 方差', 24, BLUE, a3)
    txt(gd, sd, (640, 660), '一阶可加模型 → 可复现经验常数', 22, YELLOW, a3)
    return seg(p, 0.80, 0.92)

def scene6(gd, sd, p):
    a1 = seg(p, 0.00, 0.28); a2 = seg(p, 0.26, 0.55); a3 = seg(p, 0.55, 0.82)
    txt(gd, sd, (640, 225), 'TDM: training-free 提示生成', 30, GREEN, a1)
    steps = ['① BERT 分词 → 七轴赋值', '② δ 阈值剪枝', '③ 约束组装 (≤n_c, ≤2 修饰词)']
    for i, s in enumerate(steps):
        y = 320 + i*80
        rect(gd, sd, 260, y-26, 1020, y+26, BLUE if i < 2 else GREEN, a2 if i < 2 else a3, 3)
        txt(gd, sd, (640, y), s, 26, WHITE if i < 2 else GREEN, a2 if i < 2 else a3)
    txt(gd, sd, (640, 600), '少量 GT 框离线打分, 无梯度更新', 26, YELLOW, a3)
    return seg(p, 0.78, 0.92)

def scene7(gd, sd, p):
    a1 = seg(p, 0.00, 0.25); a2 = seg(p, 0.22, 0.50); a3 = seg(p, 0.50, 0.78)
    txt(gd, sd, (640, 210), '⑦ 效果: DIOR (GDINO-tiny)', 30, BLUE, a1)
    bx0, by, bw, bh = 280, 290, 580, 54
    txt(gd, sd, (bx0, by-28), 'Detection Rate', 24, WHITE, a2)
    frect(gd, sd, bx0, by, bx0+int(bw*0.8350), by+bh, GREY, a2*0.6)
    rect(gd, sd, bx0, by, bx0+int(bw*0.8350), by+bh, GREY, a2, 3)
    txt(gd, sd, (bx0+int(bw*0.8350)+12, by+bh/2), '0.8350', 22, WHITE, a2)
    by2 = by+92
    frect(gd, sd, bx0, by2, bx0+int(bw*0.9075), by2+bh, GREEN, a3*0.8)
    rect(gd, sd, bx0, by2, bx0+int(bw*0.9075), by2+bh, GREEN, a3, 3)
    txt(gd, sd, (bx0+int(bw*0.9075)+12, by2+bh/2), '0.9075 (+0.0725)', 22, GREEN, a3)
    arrow(gd, sd, (bx0+int(bw*0.8350), by+bh/2), (bx0+int(bw*0.9075), by2+bh/2), GREEN, a3, 4)
    txt(gd, sd, (640, 560), 'F1 +0.0084 · p=0.10 未达统计显著', 26, YELLOW, a3)
    return seg(p, 0.74, 0.92)

def scene8(gd, sd, p):
    a1 = seg(p, 0.00, 0.25); a2 = seg(p, 0.22, 0.55); a3 = seg(p, 0.52, 0.80)
    txt(gd, sd, (640, 220), '⑧ 跨域 COCO: DIOR→COCO', 30, BLUE, a1)
    txt(gd, sd, (640, 335), 'ΔF1 = +0.0658', 56, GREEN, a2)
    txt(gd, sd, (640, 425), 'p < 0.001  显著', 40, GREEN, a3)
    txt(gd, sd, (640, 560), '(held-out 子集 ΔF1 = +0.0716)', 24, GREY, a3)
    return seg(p, 0.76, 0.92)

def scene9(gd, sd, p):
    a1 = seg(p, 0.00, 0.25); a2 = seg(p, 0.22, 0.55); a3 = seg(p, 0.52, 0.80)
    txt(gd, sd, (640, 210), '⑨ 效率: token 利用率', 30, WHITE, a1)
    bx0, by, bw, bh = 280, 300, 580, 54
    txt(gd, sd, (bx0, by-28), '8-token TDM', 24, GREEN, a2)
    frect(gd, sd, bx0, by, bx0+int(bw*0.830), by+bh, GREEN, a2*0.8)
    rect(gd, sd, bx0, by, bx0+int(bw*0.830), by+bh, GREEN, a2, 3)
    txt(gd, sd, (bx0+int(bw*0.830)+12, by+bh/2), 'DR 0.830', 22, GREEN, a2)
    by2 = by+92
    txt(gd, sd, (bx0, by2-28), '36-token GPT-4', 24, RED, a3)
    frect(gd, sd, bx0, by2, bx0+int(bw*0.585), by2+bh, RED, a3*0.6)
    rect(gd, sd, bx0, by2, bx0+int(bw*0.585), by2+bh, RED, a3, 3)
    txt(gd, sd, (bx0+int(bw*0.585)+12, by2+bh/2), 'DR 0.585', 22, RED, a3)
    txt(gd, sd, (640, 575), '更少 token → 更高检测率: 6.2× 利用率', 28, YELLOW, a3)
    return seg(p, 0.76, 0.92)

def scene10(gd, sd, p):
    a1 = seg(p, 0.00, 0.28); a2 = seg(p, 0.26, 0.60); a3 = seg(p, 0.58, 0.85)
    txt(gd, sd, (640, 220), '⑩ 小结', 30, WHITE, a1)
    txt(gd, sd, (640, 312), '排名 near-best: 0.2148 vs 0.2162', 28, WHITE, a2)
    txt(gd, sd, (640, 356), '(仅次几何模板, 无需逐类手调)', 22, GREY, a2)
    txt(gd, sd, (640, 440), 'TDM-Lite 跨模型: OWLv2 0.2774 → 0.3714', 28, GREEN, a3)
    txt(gd, sd, (640, 560), 'training-free · label-light · 跨模型可迁移', 26, YELLOW, a3)
    return seg(p, 0.80, 0.92)

# (时长s, 荧光绘制函数, 顶部标题, 标题色)
SCENES = [
    (10, scene1, '① 问题: 遥感OVD 提示敏感性',        BLUE),
    (13, scene2, '② 信息瓶颈: [CLS] vs 跨注意力',      BLUE),
    (12, scene3, '③ S-N 供需: 最优 token 数',           YELLOW),
    (10, scene4, '④ 条件语义容量上界',                 PURPLE),
    (8,  scene5, '⑤ OFAT 七轴语义测量',                GREEN),
    (8,  scene6, '⑥ TDM: training-free 生成',           GREEN),
    (24, scene7, '⑦ 效果: DIOR',                       BLUE),
    (24, scene8, '⑧ 效果: 跨域 COCO',                  BLUE),
    (24, scene9, '⑨ 效率: token 利用率',               WHITE),
    (22, scene10,'⑩ 小结',                            WHITE),
]
SUBS = [
    '遥感小目标+低信噪比, 提示词写法强烈影响检测',
    '静态[CLS]池化经 DPI 丢失互信息, 跨注意力保留',
    'S-N 供需模型预言 n_c≈7–12, 约束 ≤2 修饰词',
    '条件语义容量 C=7.9bit, TDM 达 42–58%',
    'OFAT 七轴一阶可加, 经验常数可复现 (PCA 84.2%)',
    'BERT分词→七轴赋值→δ剪枝→约束组装, 无梯度',
    'DIOR: DR 0.835→0.908, 域内提升为正但未显著',
    '跨域 COCO: ΔF1 +0.066, p<0.001 显著',
    '8-token TDM vs 36-token GPT-4 → 6.2× 利用率',
    'near-best 排名, 跨模型可迁移, 无需逐类手调',
]
TOTAL = sum(s[0] for s in SCENES)

def render(start=0, end=None):
    total_frames = int(TOTAL * FPS)
    if end is None:
        end = total_frames
    spans, t0 = [], 0.0
    for dur, *_ in SCENES:
        spans.append((int(t0*FPS), int((t0+dur)*FPS)))
        t0 += dur
    blur = ImageFilter.GaussianBlur(7)
    for fi in range(start, end):
        si = next(i for i, (a, b) in enumerate(spans) if a <= fi < b)
        dur, fn, title, tcol = SCENES[si]
        a, b = spans[si]
        p = (fi - a) / (b - a)
        img = Image.new('RGB', (W, H), BG)
        glow = Image.new('RGB', (W, H), (0, 0, 0))
        sharp = Image.new('RGB', (W, H), (0, 0, 0))
        gd = ImageDraw.Draw(glow); sd = ImageDraw.Draw(sharp)
        sub_a = fn(gd, sd, p)
        if title:
            txt(gd, sd, (640, 55), title, 34, tcol, seg(p, 0.0, 0.12))
        img = ImageChops.screen(img, glow.filter(blur))
        img = ImageChops.screen(img, sharp)
        d = ImageDraw.Draw(img)
        if sub_a > 0.02:
            d.text((640, 662), SUBS[si], font=font(34, True), fill=dim(WHITE, sub_a),
                   anchor='mm', stroke_width=4, stroke_fill=(0, 0, 0))
        img.save(os.path.join(OUT, 'f_%05d.png' % fi), compress_level=1)
        if fi % 400 == 0:
            print('frame %d / %d' % (fi, total_frames), flush=True)
    print('done chunk %d-%d -> %s' % (start, end, OUT), flush=True)

if __name__ == '__main__':
    import sys
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    b = int(sys.argv[2]) if len(sys.argv) > 2 else None
    render(a, b)
