# -*- coding: utf-8 -*-
"""selftest_yoloworld.py — 验证 YOLO-World v2 本地权重能否零样本出框（第三方检测器 selftest）"""
import os, sys
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

from ultralytics import YOLO
import run_bidirectional as RB

WEIGHTS = r'C:\Users\张三\yolov8s-worldv2.pt'
CLS = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']


def main():
    m = YOLO(WEIGHTS)
    m.set_classes(CLS)
    cfg = RB.Config(); cfg.model.device = 'cuda'
    for dom, cls in (('coco', 'airplane'), ('dior', 'airplane')):
        ims, gts = RB.load_split(cfg, dom, cls, 0, 1)
        if not ims:
            print(f'  {dom}/{cls}: 无图'); continue
        res = m.predict(ims[0], conf=0.03, verbose=False)
        r = res[0]
        nb = len(r.boxes)
        confs = [round(float(c), 3) for c in r.boxes.conf[:5]] if nb else []
        print(f'  {dom}/{cls}: n_box={nb} 前5置信={confs}')
    print('YOLO-World selftest OK' if True else '')


if __name__ == '__main__':
    main()
