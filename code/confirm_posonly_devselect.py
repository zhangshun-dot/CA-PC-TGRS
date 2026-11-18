# -*- coding: utf-8 -*-
r"""confirm_posonly_devselect.py — S1: positive-only 自选 beta 的留出净增量对照

外审（19:31 版）中心要求：
  让 positive-only($\lambda{=}0$) 在**开发半集**上自行选 beta，完整方法按自身
  参数规则选参，然后在**同一留出图像、同一候选池**上报告两者净差与配对 bootstrap。

复用 confirm_devtest_global_split.py 的切分与评估机制（同一批缓存，不重跑检测器）。

用法:
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u confirm_posonly_devselect.py --splits 5 --boot 1000
"""
import os, sys, json, argparse, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import numpy as np

import confirm_devtest_global_split as C

DATA = C.DATA
OUT = DATA / 'confirm_posonly_devselect.json'

# positive-only 臂的扩展 beta：主网格上限为 4.0，若 pos-only 偏好边界值，
# 需检查它是否被网格上限"人为削弱"（对称臂没有这个问题，因为它会退化）。
POS_BETAS_EXT = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def dev_posonly_scores(blobs, per_cls, betas):
    """dev 上 lam=0 行（positive-only）的六类均值增益，beta 取给定集合。"""
    base_cache = {}
    for cls, blob in blobs.items():
        pc = per_cls[cls]
        base_cache[cls] = C.ap50_fast(C.base_contribs(blob, pc['dev_idx']), pc['dev_ngt'])
    out = {}
    for be in betas:
        gs = []
        for cls, blob in blobs.items():
            pc = per_cls[cls]
            W = C.make_W(blob, be, 0.0)
            a = C.ap50_fast([C.image_contrib(blob['per_img'][i]['box'],
                                             blob['per_img'][i]['gd'],
                                             blob['per_img'][i]['gt'],
                                             blob['per_img'][i]['crop_idx'],
                                             blob['F'], W) for i in pc['dev_idx']],
                            pc['dev_ngt'])
            gs.append(a - base_cache[cls])
        out[be] = float(np.mean(gs))
    return out


def dev_pair_scores(blobs, per_cls):
    """dev 半集上的完整 4x5 网格增益（每格 = 六类均值）。"""
    scores = {}
    base_cache = {}
    for cls, blob in blobs.items():
        pc = per_cls[cls]
        base_cache[cls] = C.ap50_fast(C.base_contribs(blob, pc['dev_idx']), pc['dev_ngt'])
    for be in C.BETA_GRID:
        for la in C.LAM_GRID:
            gs = []
            for cls, blob in blobs.items():
                pc = per_cls[cls]
                W = C.make_W(blob, be, la)
                a = C.ap50_fast([C.image_contrib(blob['per_img'][i]['box'],
                                                 blob['per_img'][i]['gd'],
                                                 blob['per_img'][i]['gt'],
                                                 blob['per_img'][i]['crop_idx'],
                                                 blob['F'], W) for i in pc['dev_idx']],
                                pc['dev_ngt'])
                gs.append(a - base_cache[cls])
            scores[(be, la)] = float(np.mean(gs))
    return scores


def test_gain(blobs, per_cls, beta, lam):
    gs, per = [], {}
    for cls, blob in blobs.items():
        pc = per_cls[cls]
        b = C.ap50_fast(C.base_contribs(blob, pc['test_idx']), pc['test_ngt'])
        W = C.make_W(blob, beta, lam)
        a = C.ap50_fast([C.image_contrib(blob['per_img'][i]['box'],
                                         blob['per_img'][i]['gd'],
                                         blob['per_img'][i]['gt'],
                                         blob['per_img'][i]['crop_idx'],
                                         blob['F'], W) for i in pc['test_idx']],
                        pc['test_ngt'])
        gs.append(a - b)
        per[cls] = {'base_ap': float(b), 'gain': float(a - b)}
    return float(np.mean(gs)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()

    t0 = time.time()
    protos = C.build_protos()
    blobs = {c: C.build_or_load(c, protos) for c, _ in C.DEV_CLASSES}

    all_ids = np.unique(np.concatenate([b['img_ids'][C.K_CALIB:] for b in blobs.values()]))
    print(f'[切分] 唯一 COCO image id = {len(all_ids)} 个 (论文口径 930；pairs 合计 1291)')

    splits_out = []
    for seed in range(args.splits):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_ids))
        dev_ids = set(all_ids[perm[:len(perm) // 2]].tolist())
        test_ids = set(all_ids[perm[len(perm) // 2:]].tolist())
        assert not (dev_ids & test_ids)

        per_cls = {}
        for cls, blob in blobs.items():
            dev_mask = np.array([int(i) in dev_ids for i in blob['img_ids']])
            d_idx, d_ngt = C.split_indices(blob, dev_mask)
            t_idx, t_ngt = C.split_indices(blob, ~dev_mask)
            per_cls[cls] = {'dev_idx': d_idx, 'dev_ngt': d_ngt,
                            'test_idx': t_idx, 'test_ngt': t_ngt}

        scores = dev_pair_scores(blobs, per_cls)
        # 完整方法：全网格 dev argmax（论文同规则）
        sel_pair = max(scores, key=lambda k: scores[k])
        # positive-only：只在 lam=0 行上自行选 beta（先按主网格，再按扩展网格）
        pos_grid = dev_posonly_scores(blobs, per_cls, list(C.BETA_GRID))
        pos_ext = dev_posonly_scores(blobs, per_cls, POS_BETAS_EXT)
        pos_beta = max(C.BETA_GRID, key=lambda be: pos_grid[be])
        pos_beta_ext = max(POS_BETAS_EXT, key=lambda be: pos_ext[be])
        pos_pair = (pos_beta, 0.0)
        pos_pair_ext = (pos_beta_ext, 0.0)

        tg_sel, per_sel = test_gain(blobs, per_cls, *sel_pair)
        tg_pos, per_pos = test_gain(blobs, per_cls, *pos_pair)
        tg_pos_ext, per_pos_ext = test_gain(blobs, per_cls, *pos_pair_ext)
        # 对照：固定配置下的一对（论文既有消融的口径）
        tg_main, per_main = test_gain(blobs, per_cls, 2.0, 0.5)
        tg_mainpos, per_mainpos = test_gain(blobs, per_cls, 2.0, 0.0)

        # ── 配对 bootstrap：同一 master seed ⇒ 同一重采样 ⇒ 逐 replicate 相减 ──
        bs, _ = C.bootstrap_mean_gain(
            C.test_items_all(blobs, per_cls, *sel_pair), B=args.boot, seed=1000 + seed)
        bp, _ = C.bootstrap_mean_gain(
            C.test_items_all(blobs, per_cls, *pos_pair), B=args.boot, seed=1000 + seed)
        diff = (bs - bp) if (bs is not None and bp is not None) else None
        bpe, _ = C.bootstrap_mean_gain(
            C.test_items_all(blobs, per_cls, *pos_pair_ext), B=args.boot, seed=1000 + seed)
        diff_ext = (bs - bpe) if (bs is not None and bpe is not None) else None

        n_dev = sum(len(per_cls[c]['dev_idx']) for c in blobs)
        n_test = sum(len(per_cls[c]['test_idx']) for c in blobs)
        assert n_dev + n_test == 1291, f'dev+test={n_dev+n_test}'

        rec = {
            'seed': seed, 'n_dev_img': n_dev, 'n_test_img': n_test,
            'dev_selected_full': [float(sel_pair[0]), float(sel_pair[1])],
            'dev_gain_full': scores[sel_pair],
            'dev_selected_posonly': [float(pos_pair[0]), 0.0],
            'dev_gain_posonly': scores[pos_pair],
            'test_gain_full': tg_sel, 'test_per_class_full': per_sel,
            'test_gain_posonly': tg_pos, 'test_per_class_posonly': per_pos,
            'test_net_gain_devselected': float(tg_sel - tg_pos),
            'dev_selected_posonly_ext': [float(pos_pair_ext[0]), 0.0],
            'dev_posonly_grid': {str(k): v for k, v in pos_grid.items()},
            'dev_posonly_grid_ext': {str(k): v for k, v in pos_ext.items()},
            'test_gain_posonly_ext': tg_pos_ext, 'test_per_class_posonly_ext': per_pos_ext,
            'test_net_gain_devselected_ext': float(tg_sel - tg_pos_ext),
            'boot_net_diff_paired_ext': C.ci_stats(diff_ext),
            'test_gain_2.0_0.5': tg_main, 'test_per_class_2.0_0.5': per_main,
            'test_gain_2.0_0.0': tg_mainpos, 'test_per_class_2.0_0.0': per_mainpos,
            'test_net_gain_fixed': float(tg_main - tg_mainpos),
            'boot_full': C.ci_stats(bs),
            'boot_posonly': C.ci_stats(bp),
            'boot_net_diff_paired': C.ci_stats(diff),   # 配对差：full − pos-only
            'dev_gain_grid': {f'{b}_{l}': v for (b, l), v in scores.items()},
        }
        splits_out.append(rec)

        bt = C.ci_stats(diff)
        print(f'\n[split {seed}] dev={n_dev} / test={n_test}')
        print(f'  dev: full={sel_pair} ({scores[sel_pair]:+.4f})   pos-only={pos_pair} ({scores[pos_pair]:+.4f})')
        print(f'  test: full={tg_sel:+.4f}   pos-only={tg_pos:+.4f}   net(full-pos)={tg_sel - tg_pos:+.4f}')
        print(f'  pos-only 扩展 beta: dev 选 {pos_pair_ext} -> test={tg_pos_ext:+.4f}  '
              f'net(full-pos_ext)={tg_sel - tg_pos_ext:+.4f}')
        print(f'  dev 上 lam=0 曲线: {[(k, round(v, 4)) for k, v in pos_ext.items()]}')
        print(f'  test(固定口径): (2.0,0.5)={tg_main:+.4f}  (2.0,0.0)={tg_mainpos:+.4f}  net={tg_main - tg_mainpos:+.4f}')
        if bt:
            print(f'  paired bootstrap net: mean={bt["mean"]:+.4f} '
                  f'95%CI=[{bt["lo95"]:+.4f},{bt["hi95"]:+.4f}] '
                  f'非正={bt["n_le_0"]}/{bt["n_boot"]}')

    nets = [r['test_net_gain_devselected'] for r in splits_out]
    nets_ext = [r['test_net_gain_devselected_ext'] for r in splits_out]
    fixed = [r['test_net_gain_fixed'] for r in splits_out]
    pos_betas = [r['dev_selected_posonly'][0] for r in splits_out]
    pos_betas_ext = [r['dev_selected_posonly_ext'][0] for r in splits_out]
    summary = {
        'n_splits': len(splits_out),
        'test_net_gain_devselected_mean': float(np.mean(nets)),
        'test_net_gain_devselected_min': float(np.min(nets)),
        'test_net_gain_devselected_max': float(np.max(nets)),
        'test_net_gain_devselected_ext_mean': float(np.mean(nets_ext)),
        'test_net_gain_devselected_ext_min': float(np.min(nets_ext)),
        'test_net_gain_devselected_ext_max': float(np.max(nets_ext)),
        'test_gain_full_mean': float(np.mean([r['test_gain_full'] for r in splits_out])),
        'test_gain_posonly_mean': float(np.mean([r['test_gain_posonly'] for r in splits_out])),
        'test_gain_posonly_ext_mean': float(np.mean([r['test_gain_posonly_ext'] for r in splits_out])),
        'test_net_gain_fixed_mean': float(np.mean(fixed)),
        'posonly_selected_betas': pos_betas,
        'posonly_selected_betas_ext': pos_betas_ext,
        'full_selected_pairs': [r['dev_selected_full'] for r in splits_out],
        'splits_with_paired_CI_excluding_0': int(sum(
            1 for r in splits_out
            if r['boot_net_diff_paired'] and r['boot_net_diff_paired']['lo95'] > 0)),
        'splits_with_paired_CI_excluding_0_ext': int(sum(
            1 for r in splits_out
            if r['boot_net_diff_paired_ext'] and r['boot_net_diff_paired_ext']['lo95'] > 0)),
        'boot_net_paired_nonpositive_counts': [
            (r['boot_net_diff_paired']['n_le_0'], r['boot_net_diff_paired']['n_boot'])
            for r in splits_out if r['boot_net_diff_paired']],
        'boot_net_paired_ext_nonpositive_counts': [
            (r['boot_net_diff_paired_ext']['n_le_0'], r['boot_net_diff_paired_ext']['n_boot'])
            for r in splits_out if r['boot_net_diff_paired_ext']],
    }
    print('\n==== 汇总 ====')
    for k, v in summary.items():
        print(f'  {k}: {v}')

    out = {'config': {
        'detector': 'gdino-tiny', 'clip': 'ViT-B/32 (openai)',
        'K_CALIB': C.K_CALIB, 'pre_filter': C.PRE_FILTER,
        'beta_grid': C.BETA_GRID, 'lam_grid': C.LAM_GRID,
        'split': 'COCO image ID 全局二等分, dev/test 不相交',
        'full_rule': 'dev 全网格 argmax(六类均值增益)',
        'posonly_rule': 'dev 上 lam=0 行 argmax(beta)',
        'posonly_betas_ext': POS_BETAS_EXT,
        'paired_bootstrap': f'同 master seed -> 同一图重采样 -> full−posonly 逐 replicate 相减 (B={args.boot})',
        'elapsed_s': round(time.time() - t0, 1),
    }, 'summary': summary, 'splits': splits_out}
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
