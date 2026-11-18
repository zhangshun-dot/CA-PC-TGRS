# -*- coding: utf-8 -*-
"""aic_principle_video.py — A-CSPC 原理段视频 (黑底荧光板书风)

风格对齐抖音科普模板: 黑底 + 荧光手绘线条(glow) + 底部白色大字幕, 快节奏。
6 幕 / 共 58s / 1280x720 / 25fps, 帧序列输出后由 ffmpeg 合成, 再与效果段拼接。

运行: C:/Users/张三/.workbuddy/binaries/python/envs/default/Scripts/python.exe aic_principle_video.py
"""
import os, math
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageFont

W, H = 1280, 720
FPS = 25
OUT = r'C:\Users\张三\Desktop\实验数据\_principle_frames'
os.makedirs(OUT, exist_ok=True)

FONT_B = r'C:\Windows\Fonts\msyhbd.ttc'
FONT_R = r'C:\Windows\Fonts\msyh.ttc'
_fonts = {}
def font(sz, bold=True):
    k = (sz, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT_R, sz)
    return _fonts[k]

# 荧光配色
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

# ─────────────── 荧光绘制原语 (gd=glow层, sd=清晰层) ───────────────
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

# ─────────────── 图标 ───────────────
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

def drone(gd, sd, c, color, a):
    x, y = c
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        ex, ey = x+dx*60, y+dy*60
        L(gd, sd, (x, y), (ex, ey), color, a, 4)
        ring(gd, sd, (ex, ey), 15, color, a, 3)
    rect(gd, sd, x-16, y-9, x+16, y+9, color, a, 4)

def padlock(gd, sd, c, color, a):
    x, y = c
    rect(gd, sd, x-16, y-6, x+16, y+18, color, a, 3)
    gd.arc([x-11, y-20, x+11, y+2], 180, 360, fill=dim(color, a), width=4)
    sd.arc([x-11, y-20, x+11, y+2], 180, 360, fill=dim(color, a), width=3)

# ─────────────── 六幕 ───────────────
def scene1(gd, sd, p):
    a1 = seg(p, 0.00, 0.22); a2 = seg(p, 0.18, 0.40); a3 = seg(p, 0.42, 0.68)
    satellite(gd, sd, (250, 290), BLUE, a1)
    txt(gd, sd, (250, 420), 'DIOR 源域', 32, WHITE, a1)
    txt(gd, sd, (250, 462), '卫星视角', 26, GREY, a1)
    drone(gd, sd, (1030, 290), GREEN, a2)
    txt(gd, sd, (1030, 420), 'VisDrone 目标域', 32, WHITE, a2)
    txt(gd, sd, (1030, 462), '无人机低空', 26, GREY, a2)
    pts = [(430, 250), (560, 320), (490, 375), (640, 445), (575, 500), (700, 560)]
    for i in range(len(pts)-1):
        L(gd, sd, pts[i], pts[i+1], YELLOW, a3, 5)
    txt(gd, sd, (640, 200), '视角 · 尺度 剧变', 30, YELLOW, a3)
    return seg(p, 0.62, 0.85)

def scene2(gd, sd, p):
    a_sp = seg(p, 0.00, 0.16); a_src = seg(p, 0.13, 0.30); a_ar = seg(p, 0.28, 0.50)
    a_hat = seg(p, 0.46, 0.64); a_fm = seg(p, 0.60, 0.78)
    dashed_ring(gd, sd, (640, 410), 390, 180, GREY, a_sp, 3)
    txt(gd, sd, (640, 250), 'CLIP 共享空间', 26, GREY, a_sp)
    dot(gd, sd, (430, 480), GREEN, a_src, 11)
    txt(gd, sd, (395, 528), '源原型 P_src', 25, GREEN, a_src)
    arrow(gd, sd, (448, 465), (742, 322), YELLOW, a_ar, 5)
    txt(gd, sd, (545, 355), '文本方向 Δ', 25, YELLOW, a_ar)
    dot(gd, sd, (762, 308), BLUE, a_hat, 11)
    txt(gd, sd, (812, 262), '外推原型 P′', 25, BLUE, a_hat)
    txt(gd, sd, (640, 150), 'P′ = norm( P_src + β·Δ )    β = 2.0', 30, YELLOW, a_fm)
    return seg(p, 0.70, 0.92)

def scene3(gd, sd, p):
    aL = seg(p, 0.00, 0.28); aR = seg(p, 0.38, 0.70)
    L(gd, sd, (640, 115), (640, 605), (70, 70, 80), 1.0, 2)
    # 左: A-CSPC 非对称
    txt(gd, sd, (330, 78), 'A-CSPC · 非对称', 33, GREEN, aL)
    L(gd, sd, (555, 66), (582, 94), GREEN, aL, 5)
    L(gd, sd, (582, 94), (625, 42), GREEN, aL, 5)
    dot(gd, sd, (230, 470), GREEN, aL, 11)
    txt(gd, sd, (215, 518), '正原型', 24, GREEN, aL)
    arrow(gd, sd, (246, 458), (348, 372), GREEN, aL, 4)
    txt(gd, sd, (430, 336), '外推 → 目标域', 24, GREEN, aL)
    dot(gd, sd, (500, 545), PURPLE, aL, 11)
    padlock(gd, sd, (500, 585), PURPLE, aL)
    txt(gd, sd, (500, 632), '负原型 · 留在源域', 24, PURPLE, aL)
    # 右: 对称外推对照
    txt(gd, sd, (950, 78), '对称外推 · 强对照', 33, RED, aR)
    L(gd, sd, (852, 52), (888, 94), RED, aR, 5)
    L(gd, sd, (888, 52), (852, 94), RED, aR, 5)
    dot(gd, sd, (790, 470), GREEN, aR, 11)
    arrow(gd, sd, (806, 458), (902, 378), GREEN, aR, 4)
    dot(gd, sd, (990, 545), PURPLE, aR, 11)
    dash(gd, sd, (1006, 533), (1098, 452), PURPLE, aR, 4)
    arrow(gd, sd, (1080, 464), (1098, 450), PURPLE, aR, 4)
    txt(gd, sd, (990, 592), '负原型也外推', 24, PURPLE, aR)
    txt(gd, sd, (950, 300), '类间抑制失效 → 翻车', 25, RED, aR)
    return seg(p, 0.78, 0.95)

def scene4(gd, sd, p):
    a1 = seg(p, 0.00, 0.16); a2 = seg(p, 0.12, 0.28); a3 = seg(p, 0.26, 0.46)
    a4 = seg(p, 0.44, 0.62)
    rect(gd, sd, 120, 310, 440, 400, BLUE, a1, 4)
    txt(gd, sd, (280, 355), '检测器分数 gd', 28, WHITE, a1)
    rect(gd, sd, 120, 460, 440, 585, GREEN, a2, 4)
    txt(gd, sd, (280, 505), 'CLIP 视觉证据', 28, WHITE, a2)
    txt(gd, sd, (280, 550), 'vis = max(0, pos - λ·neg)', 22, GREEN, a2)
    arrow(gd, sd, (445, 355), (622, 428), BLUE, a3, 4)
    arrow(gd, sd, (445, 520), (622, 452), GREEN, a3, 4)
    ring(gd, sd, (700, 440), 52, YELLOW, a3, 5)
    txt(gd, sd, (700, 438), '√', 44, YELLOW, a3)
    arrow(gd, sd, (760, 440), (862, 440), YELLOW, a4, 4)
    rect(gd, sd, 870, 370, 1190, 510, YELLOW, a4, 4)
    txt(gd, sd, (1030, 420), 'fused =', 26, WHITE, a4)
    txt(gd, sd, (1030, 468), '√( gd × vis )', 30, YELLOW, a4)
    return seg(p, 0.70, 0.92)

def scene5(gd, sd, p):
    a_ax = seg(p, 0.00, 0.14); a_tau = seg(p, 0.15, 0.34)
    a_rg = seg(p, 0.35, 0.58); a_nt = seg(p, 0.58, 0.76)
    arrow(gd, sd, (140, 470), (1140, 470), GREY, a_ax, 4)
    txt(gd, sd, (140, 512), '0', 24, GREY, a_ax)
    txt(gd, sd, (1060, 515), 'g2 对齐信号', 26, GREY, a_ax)
    dash(gd, sd, (430, 300), (430, 468), YELLOW, a_tau, 4)
    txt(gd, sd, (430, 268), 'τ* = 0.174', 30, YELLOW, a_tau)
    txt(gd, sd, (285, 355), 'g2 < τ*', 30, GREY, a_rg)
    txt(gd, sd, (285, 402), '退回 Baseline', 26, BLUE, a_rg)
    txt(gd, sd, (795, 355), 'g2 ≥ τ*', 30, GREEN, a_rg)
    txt(gd, sd, (795, 402), '采用 A-CSPC 校正', 26, GREEN, a_rg)
    txt(gd, sd, (640, 562), '22 个类-域点校准 · 最差回归 -0.081 → -0.010', 24, WHITE, a_nt)
    return seg(p, 0.74, 0.92)

def scene6(gd, sd, p):
    a1 = seg(p, 0.00, 0.35); a2 = seg(p, 0.28, 0.58)
    txt(gd, sd, (640, 320), '真实效果如何?', 60, WHITE, a1)
    txt(gd, sd, (640, 425), 'DIOR→VisDrone · COCO 6类 · 11点 AP50', 28, GREY, a2)
    return seg(p, 0.50, 0.80)

# (时长s, 荧光绘制函数, 顶部标题, 标题色, 字幕)
SCENES = [
    (9,  scene1, '① 问题:跨域鸿沟',        BLUE),
    (11, scene2, '② CLIP 原型外推',         BLUE),
    (12, scene3, '③ 非对称 · 核心',         PURPLE),
    (10, scene4, '④ 融合打分',              GREEN),
    (10, scene5, '⑤ Safe-Correction 门控',  YELLOW),
    (6,  scene6, '',                       WHITE),
]
SUBS = [
    '训练-free 跨域检测:卫星源域 → 无人机低空目标域',
    'CLIP 共享空间里,把源原型沿文本方向外推到目标域',
    '非对称才是核心:负原型必须留在源域不动',
    '几何平均融合检测器分数与 CLIP 视觉证据,只改排序',
    'g2 对齐信号做安全门控:拿不准就不校正',
    '接下来:真实数据上的效果',
]
TOTAL = sum(s[0] for s in SCENES)

def render():
    total_frames = int(TOTAL * FPS)
    # 预计算每幕的帧区间
    spans, t0 = [], 0.0
    for dur, *_ in SCENES:
        spans.append((int(t0*FPS), int((t0+dur)*FPS)))
        t0 += dur
    blur = ImageFilter.GaussianBlur(7)
    for fi in range(total_frames):
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
        if fi % 250 == 0:
            print('frame %d / %d' % (fi, total_frames), flush=True)
    print('done: %d frames -> %s' % (total_frames, OUT), flush=True)

if __name__ == '__main__':
    render()
