# -*- coding: utf-8 -*-
"""aic_demo_video.py — 比赛演示视频 (3-5 分钟 MP4)
VisDrone 低空目标域: baseline(红) vs A-CSPC(蓝) 逐框淡入对比动画 + 中文解说卡。
复用检测缓存, 仅 CLIP 编码(约 5 分钟) + 渲染(后台)。
"""
import sys, os, json, importlib.util
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2

import aic_visdrone_experiment as V

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'aic_demo_video.mp4'
FONT = r'C:\Windows\Fonts\msyh.ttc'

BETA, LAM = V.BETA, V.LAM
SRC_BANK = V.SRC_BANK
SRC_OF = V.SRC_OF
FPS = 25
W, H = 1280, 720


def font(sz):
    return ImageFont.truetype(FONT, sz)


def scale_to_canvas(img, boxes):
    """letterbox 到 1280x720, 返回 (PIL, scale, ox, oy)。boxes 同步缩放。"""
    w, h = img.size
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    canvas = Image.new('RGB', (W, H), (0, 0, 0))
    px, py = (W - nw) // 2, (H - nh) // 2
    canvas.paste(img.resize((nw, nh)), (px, py))
    sb = []
    for b in boxes:
        sb.append([b[0] * s + px, b[1] * s + py, b[2] * s + px, b[3] * s + py])
    return canvas, sb


def greedy_match(boxes, scores, gts, iou_th=0.5):
    """返回 (matched_count, per_box_matched_bool)。"""
    if len(gts) == 0 or len(boxes) == 0:
        return 0, [False] * len(boxes)
    order = np.argsort(-np.array(scores))
    gts = np.array(gts, dtype=float)
    matched = [False] * len(gts)
    bm = [False] * len(boxes)
    for i in order:
        bi = boxes[i]
        best, bj = 0.0, -1
        for j in range(len(gts)):
            if matched[j]:
                continue
            ix1, iy1, ix2, iy2 = bi
            jx1, jy1, jx2, jy2 = gts[j]
            iw = max(0, min(ix2, jx2) - max(ix1, jx1))
            ih = max(0, min(iy2, jy2) - max(iy1, jy1))
            inter = iw * ih
            area = (ix2 - ix1) * (iy2 - iy1) + (jx2 - jx1) * (jy2 - jy1) - inter
            iou = inter / max(area, 1e-9)
            if iou > best:
                best, bj = iou, j
        if best >= iou_th:
            matched[bj] = True; bm[i] = True
    return int(sum(matched)), bm


def build_clips():
    """对每个类: 计算逐图 fused, 选 top-K 张「A-CSPC 新增召回」图。"""
    print('=== 源域原型 ===')
    src = {}
    for c in SRC_BANK:
        images, gts = V.U.load_dior(c, n=50)
        crops = []
        for img, g in zip(images, gts):
            for gt in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 4 or y2 <= y1 + 4:
                    continue
                crops.append(img.crop((x1, y1, x2, y2)))
        F = V.T.encode_clip_batch(crops).cpu().numpy()
        p = F.mean(axis=0)
        src[c] = {'proto': p / (np.linalg.norm(p) + 1e-8), 'features': F}
        print(f'  {c}: {len(crops)} crops')

    gt_all = V.load_visdrone_gt()
    all_src = np.vstack([src[c]['features'] for c in SRC_BANK])

    clips = []
    for coco_cls in ['bus', 'motorcycle', 'car']:
        entries = gt_all[coco_cls]
        dets = V.detect_visdrone(entries, V.TGT_TMPL.format(coco_cls))
        tgt = V.build_target(entries, dets)
        src_cls = SRC_OF[coco_cls]
        delta = V.text_delta(src_cls, coco_cls)
        neg = V.build_neg_from(src, src_cls)
        p_tge = src[src_cls]['proto'] + BETA * delta
        p_tge /= (np.linalg.norm(p_tge) + 1e-8)
        base_f = V.fuse_scores(tgt, None, None, 0)
        acspc_f = V.fuse_scores(tgt, p_tge, neg, LAM)

        scored = []
        for i, p in enumerate(tgt['per_img']):
            apb = V.img_ap(p, base_f[i], 0.5)
            apa = V.img_ap(p, acspc_f[i], 0.5)
            scored.append((apa - apb, i, len(p['gt']), apa, apb))
        scored.sort(key=lambda x: -x[0])
        top = scored[:4]
        print(f'  {coco_cls}: 选 {len(top)} 张, 逐图AP增益={[round(s[0], 3) for s in top]}')
        for gain, i, ngt, apa, apb in top:
            clips.append({'cls': coco_cls, 'idx': i, 'entries': entries,
                          'per_img': tgt['per_img'][i], 'base_f': base_f[i],
                          'acspc_f': acspc_f[i], 'ngt': ngt,
                          'apb': round(float(apb), 3), 'apa': round(float(apa), 3),
                          'apg': round(float(gain) * 100, 1)})
    return clips


def draw_frame(canvas, per_img, base_f, acspc_f, nbase_show, nacspc_show,
               phase, cls, ngt, na, nb, new):
    """在 canvas(RGB PIL)上叠加当前帧的框与文字。"""
    d = ImageDraw.Draw(canvas)
    boxes = per_img['box']; gts = per_img['gt']
    # GT 绿框(常显淡)
    for g in gts:
        d.rectangle(list(g), outline=(40, 200, 60), width=2)
    # baseline 红框(按分数前 nbase_show 个)
    bs = np.argsort(-np.array(base_f))
    for k in range(min(nbase_show, len(bs))):
        i = bs[k]
        d.rectangle(list(boxes[i]), outline=(230, 60, 60), width=3)
    # A-CSPC 蓝框(前 nacspc_show 个)
    as_ = np.argsort(-np.array(acspc_f))
    for k in range(min(nacspc_show, len(as_))):
        i = as_[k]
        d.rectangle(list(boxes[i]), outline=(60, 120, 255), width=3)
    # 文字卡
    d.text((20, 20), f'VisDrone 低空 · {cls}', font=font(34), fill=(255, 255, 255))
    d.text((20, H - 90), f'GT: {ngt}   Baseline 召回: {nb}   A-CSPC 召回: {na}   (+{new})',
           font=font(28), fill=(255, 230, 120))
    d.text((20, H - 50), '红=Baseline   蓝=A-CSPC   绿=真实框', font=font(24),
           fill=(200, 200, 200))
    return canvas


def make_title(dur=12):
    frames = []
    for f in range(dur * FPS):
        cv = Image.new('RGB', (W, H), (12, 18, 30))
        d = ImageDraw.Draw(cv)
        a = min(1.0, f / (FPS * 1.5))
        d.text((W // 2 - 380, 220), 'A-CSPC 跨域原型校正', font=font(56),
               fill=(255, 255, 255))
        d.text((W // 2 - 300, 300), '低空无人机目标检测 · 演示', font=font(40),
               fill=(120, 200, 255))
        d.text((W // 2 - 360, 420), '训练-free · 卫星源域 → 无人机低空目标域', font=font(28),
               fill=(200, 200, 200))
        frames.append(cv)
    return frames


def make_metric(dur=18):
    frames = []
    for f in range(dur * FPS):
        cv = Image.new('RGB', (W, H), (12, 18, 30))
        d = ImageDraw.Draw(cv)
        d.text((80, 120), '统一官方指标结果', font=font(48), fill=(255, 255, 255))
        lines = [
            ('DIOR→COCO (6类) A-CSPC AP50', '+0.055  (0.377→0.433)'),
            ('VisDrone bus AP50 + 显著性', '+0.111  [+0.069,+0.158] 显著'),
            ('VisDrone motorcycle AP50 + 显著性', '+0.078  [+0.068,+0.089] 显著'),
            ('VisDrone car AP50 + 显著性', '+0.013  [+0.003,+0.022] 显著'),
            ('Safe-Correction 最差回退', '−0.0808 → −0.0104'),
        ]
        y = 220
        for k, v in lines:
            d.text((90, y), k, font=font(30), fill=(220, 220, 220))
            d.text((950, y), v, font=font(30), fill=(120, 255, 160))
            y += 80
        frames.append(cv)
    return frames


def main():
    clips = build_clips()
    title = make_title()
    metric = make_metric()

    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

    def write(cv):
        writer.write(cv2.cvtColor(np.array(cv), cv2.COLOR_RGB2BGR))

    for cv in title:
        write(cv)
    print('标题卡完成', flush=True)

    img_dur = 17  # 秒/图
    nframes = img_dur * FPS
    for ci, clip in enumerate(clips):
        per_img = clip['per_img']
        base_f, acspc_f = clip['base_f'], clip['acspc_f']
        nbox = len(per_img['box'])
        # 预缩放原图 + GT 为画布底
        img = Image.open(clip['entries'][clip['idx']][0]).convert('RGB')
        canvas0, _ = scale_to_canvas(img, [])
        # 把 GT/框也缩放到画布坐标
        _, gts_s = scale_to_canvas(img, per_img['gt'])
        _, boxes_s = scale_to_canvas(img, per_img['box'])
        per_s = {'box': np.array(boxes_s), 'gt': np.array(gts_s),
                 'gd': per_img['gd'], 'crop_idx': per_img['crop_idx'],
                 'iou': None}
        # 重算缩放后的匹配与展示数量(用缩放后坐标, 阈值 .5)
        nb_shown, _ = greedy_match(per_s['box'], base_f, per_s['gt'])
        na_shown, _ = greedy_match(per_s['box'], acspc_f, per_s['gt'])
        for fr in range(nframes):
            t = fr / nframes
            # 阶段: 0-0.12 仅 GT; 0.12-0.6 baseline 逐框; 0.6-1.0 A-CSPC 逐框
            if t < 0.12:
                nbase, nacspc = 0, 0
            elif t < 0.6:
                nbase = int(min(nbox, (t - 0.12) / 0.48 * nbox))
                nacspc = 0
            else:
                nbase = nbox
                nacspc = int(min(nbox, (t - 0.6) / 0.4 * nbox))
            cv = canvas0.copy()
            d = ImageDraw.Draw(cv)
            for g in per_s['gt']:
                d.rectangle(list(g), outline=(40, 200, 60), width=2)
            bs = np.argsort(-np.array(base_f))
            for k in range(min(nbase, len(bs))):
                i = bs[k]; d.rectangle(list(per_s['box'][i]), outline=(230, 60, 60), width=3)
            as_ = np.argsort(-np.array(acspc_f))
            for k in range(min(nacspc, len(as_))):
                i = as_[k]; d.rectangle(list(per_s['box'][i]), outline=(60, 120, 255), width=3)
            d.text((20, 20), f'VisDrone 低空 · {clip["cls"]}', font=font(34),
                   fill=(255, 255, 255))
            d.text((20, H - 90), f'GT: {clip["ngt"]}   baseline AP {clip["apb"]:.3f}'
                   f' → A-CSPC {clip["apa"]:.3f}   (+{clip["apg"]}pp)', font=font(26),
                   fill=(255, 230, 120))
            d.text((20, H - 50), '红=Baseline   蓝=A-CSPC   绿=真实框', font=font(22),
                   fill=(200, 200, 200))
            write(cv)
        print(f'  图 {ci+1}/{len(clips)} {clip["cls"]} 完成', flush=True)

    for cv in metric:
        write(cv)
    writer.release()
    print(f'视频已写入 {OUT}  时长≈{(len(title)+len(clips)*nframes+len(metric))/FPS:.0f}s')


if __name__ == '__main__':
    main()
