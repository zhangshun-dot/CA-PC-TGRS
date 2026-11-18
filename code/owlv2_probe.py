# -*- coding: utf-8 -*-
"""owlv2_probe.py — 主干迁移最小探针: OWLv2 vs GDINO-tiny 的跨域 baseline

目的: 判断"换主干"值不值得继续。不跑 CDVM 方法, 只比基线（最便宜、最决定性）:
  同一协议下 OWLv2 与 GDINO-tiny 在 COCO 跨域 3 类上的 baseline AP50。

协议与 unified_eval 完全一致:  GDINO th=0 → pre-filter 0.03 → 数据集级 11-point AP50
判定阈值:
  OWLv2 baseline ≥ GDINO-tiny + 0.02  → 值得跑完整方法 (写个对照表, 不换主表)
  否则 → 锁死 GDINO-tiny, related work 一句交代

OWLv2 接口差异 (unified_eval 的 gdino_detect 需适配):
  processor: OwlViTProcessor.from_pretrained
  model:     OwlViTForObjectDetection.from_pretrained
  forward:   text=...  (非 [text, [text]])
  decode:    post_process_object_detection(outputs, threshold, target_sizes)  → logits→scores

前置: 检查 OWLv2 模型是否已缓存; 若离线且无缓存则报错退出。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

# ── OWLv2 缓存路径（若不在，则按标准 HF 缓存名推断）──
OWL_CANDIDATES = [
    r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble',
    r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16',
]
OWL_MODEL = None
for c in OWL_CANDIDATES:
    if os.path.exists(c):
        OWL_MODEL = c
        break

# ── 若缓存缺失，用标准名尝试在线加载（会失败则报错）──
OWL_HF_NAME = 'google/owlv2-base-patch16-ensemble'


def resolve_snapshot(cache_dir):
    """把 HF 缓存外壳目录解析到 snapshot 目录（refs/main 的 commit hash）。

    之前失败的根因: 把 'models--google--owlv2-base-patch16-ensemble' 外壳目录
    直接传给 from_pretrained，而 preprocessor_config.json / model.safetensors
    实际在 snapshots/<commit>/ 子目录里；from_pretrained 收到目录时只在该目录
    本身找文件，找不到就报 "can't load image processor"。
    """
    refs_dir = os.path.join(cache_dir, 'refs')
    if not os.path.isdir(refs_dir):
        return cache_dir
    for rev in sorted(os.listdir(refs_dir)):
        ref_file = os.path.join(refs_dir, rev)
        if not os.path.isfile(ref_file):
            continue
        try:
            with open(ref_file) as f:
                commit = f.read().strip()
        except OSError:
            continue
        snap = os.path.join(cache_dir, 'snapshots', commit)
        if os.path.isfile(os.path.join(snap, 'config.json')):
            return snap
    return cache_dir


def get_owl():
    if 'owl' not in U._gd:
        # 类名必须匹配 config.json 的 model_type。config 写 "owlv2" → 用 Owlv2* 类。
        # 之前 try 分支先试 OwlViT* (旧名在这版 transformers 里仍存在) → 用 owlvit 类
        # 去装 owlv2 权重, key 对不上 → 报 "state dictionary corrupted"。反过来优先 Owlv2*。
        import transformers as _tf
        if hasattr(_tf, 'Owlv2Processor'):
            OWL_Processor, OWL_Model = _tf.Owlv2Processor, _tf.Owlv2ForObjectDetection
        else:
            OWL_Processor, OWL_Model = _tf.OwlViTProcessor, _tf.OwlViTForObjectDetection

        if OWL_MODEL:
            model_name = resolve_snapshot(OWL_MODEL)
            print(f'  加载 OWLv2 (本地缓存 snapshot): {model_name}')
        else:
            model_name = OWL_HF_NAME
            print(f'  加载 OWLv2 (在线): {model_name}')
        U._gd['owl_p'] = OWL_Processor.from_pretrained(model_name)
        # config 已是 float32; 不传 torch_dtype, 避免 deprecation 告警
        U._gd['owl_m'] = OWL_Model.from_pretrained(model_name).eval().to(U.DEVICE)
    return U._gd['owl_p'], U._gd['owl_m']


def owl_detect(images, prompt, thresh=0.0):
    """OWLv2 检测: 返回每图 (boxes, scores)，协议对齐 gdino_detect。"""
    owl_p, owl_m = get_owl()
    out = []
    for img in images:
        ins = owl_p(images=img, text=[prompt], return_tensors='pt')
        ins = {k: v.to(U.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
        with torch.no_grad():
            o = owl_m(**ins)
        ts = torch.tensor([img.size[::-1]]).to(U.DEVICE)
        r = owl_p.post_process_object_detection(
            o, threshold=thresh, target_sizes=ts)[0]
        out.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return out


def baseline_ap(images, gts, prompt):
    dets = owl_detect(images, prompt, 0.0)
    all_b, all_s, all_g = [], [], []
    for (boxes, scores), g in zip(dets, gts):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        all_b.extend(boxes[keep].tolist())
        all_s.extend(scores[keep].tolist())
        all_g.extend(g)
    return U.compute_ap50(all_b, all_s, all_g)


def main():
    # GDINO-tiny 跨域基线（从 unified_eval 现有协议直接算，作对照）
    print('=' * 70)
    print('OWLv2 vs GDINO-tiny — COCO 跨域 baseline 探针')
    print(f'协议: pre-filter={U.PRE_FILTER} → 数据集级 AP50')
    print('=' * 70)

    if OWL_MODEL is None:
        print('!! 未找到 OWLv2 本地缓存，将尝试在线加载 google/owlv2-base-patch16-ensemble')
        print('!! 若离线会失败 → 直接判定锁死 GDINO-tiny')

    COCO_CLS = [
        ('COCO-airplane', 'airplane', 'a photo of', 'airplane'),
        ('COCO-boat', 'boat', 'a photo of', 'ship'),
        ('COCO-car', 'car', 'a photo of', 'vehicle'),
    ]

    rows = {}
    print(f'\n{"类":<16s} {"GDINO-tiny":>10s} {"OWLv2":>10s} {"Δ":>8s}')
    print('-' * 48)

    for name, cls, prefix, dior_cls in COCO_CLS:
        images, gts = U.load_coco(cls, n=U.N_COCO)
        print(f'\n[{name}]  {len(images)} 图')
        # GDINO baseline（复用 unified_eval）
        gd_dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)
        bb, bs, bgt = U.filter_flatten(gd_dets, gts)
        gd_ap = U.compute_ap50(bb, bs, bgt)
        # OWLv2 baseline
        try:
            owl_ap = baseline_ap(images, gts, f'{prefix} {cls}')
        except Exception as e:
            print(f'  OWLv2 失败: {e}')
            owl_ap = float('nan')
        rows[name] = {'gdino': gd_ap, 'owl': owl_ap}
        print(f'{name:<16s} {gd_ap:>10.4f} {owl_ap:>10.4f} {owl_ap - gd_ap:>+8.4f}')

    # 判定
    print('\n' + '=' * 70)
    ows = [rows[n]['owl'] for n in rows if rows[n]['owl'] == rows[n]['owl']]
    if not ows:
        print('OWLv2 全部失败 → 锁死 GDINO-tiny')
    else:
        mean_d = np.mean([rows[n]['owl'] - rows[n]['gdino'] for n in rows])
        print(f'平均 Δ(OWL−GDINO) = {mean_d:+.4f}')
        if mean_d >= 0.02:
            print('→ 值得跑完整 CDVM 对照表（写进主表旁/消融）')
        elif mean_d >= 0.0:
            print('→ 中性，锁死 GDINO-tiny，related work 一句交代')
        else:
            print('→ OWLv2 更差，锁死 GDINO-tiny')


if __name__ == '__main__':
    main()
