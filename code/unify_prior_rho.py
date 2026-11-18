# -*- coding: utf-8 -*-
"""统一先验口径：用「真正产出结果的权重」重算 rho(37 行/6 域对)。

背景：
  现稿 Eq.(risk) 印 w_G=0.20,w_S=0.15,w_I=0.35,w_C=0.30；
  但产出 alpha 自适应结果(+0.0478)的管线用的是
  PhysicsPrior.risk_score 默认权重 0.25/0.15/0.25/0.35（槽位 V,M,S,I），
  且 load_human_priors_from_dict 会把 [0,1] 值 *5 后 clamp 到 [1,5] 再 /5，
  等价于把小于 0.2 的赋值抬到 0.2。
  本脚本在同一 37 行上对比三种口径的 rho：
    pub   : 公布权重(0.20/0.15/0.35/0.30) x Table22 原始赋值（不 clamp）
    pipe  : 管线权重(0.25/0.15/0.25/0.35) x Table22 clamp 后赋值  <-- 与 alpha 结果同源
    stored: m3 产物里存的风险值（无法由任何已公布权重线性重构，仅作参照）
"""
import io, json
from scipy.stats import spearmanr

M3 = r"C:\Users\张三\Desktop\实验数据\m3_cluster_bootstrap.json"

# Table 22 (G granularity mismatch, S scale, I intra-class variation, C inter-class confusion)
T22 = {
    'airplane':          (0.65, 0.15, 0.20, 0.35),
    'boat':              (0.90, 0.70, 0.30, 0.55),
    'ship':              (0.85, 0.65, 0.25, 0.60),
    'car':               (0.50, 0.10, 0.70, 0.50),
    'truck':             (0.60, 0.30, 0.40, 0.55),
    'bus':               (0.65, 0.50, 0.40, 0.40),
    'motorcycle':        (0.95, 0.55, 0.90, 0.45),
    'storage tank':      (0.70, 0.20, 0.30, 0.45),
    'baseball diamond':  (0.40, 0.10, 0.10, 0.65),
    'tennis court':      (0.30, 0.10, 0.10, 0.25),
    'basketball court':  (0.30, 0.10, 0.10, 0.30),
    'ground track field':(0.35, 0.10, 0.05, 0.70),
    'harbor':            (0.50, 0.40, 0.15, 0.75),
    'bridge':            (0.55, 0.30, 0.20, 0.50),
    'vehicle':           (0.60, 0.20, 0.60, 0.70),
}
ALIAS = {'aeroplane': 'airplane', 'motorbike': 'motorcycle'}

W_PUB = (0.20, 0.15, 0.35, 0.30)
W_PIPE = (0.25, 0.15, 0.25, 0.35)


def clamp5(v):
    """load_human_priors_from_dict: [0,1] -> *5 -> clamp[1,5] -> /5"""
    return min(max(v * 5.0, 1.0), 5.0) / 5.0


def risk_pub(name):
    g, s, i, c = T22[name]
    return W_PUB[0]*g + W_PUB[1]*s + W_PUB[2]*i + W_PUB[3]*c


def risk_pipe(name):
    g, s, i, c = [clamp5(v) for v in T22[name]]
    return W_PIPE[0]*g + W_PIPE[1]*s + W_PIPE[2]*i + W_PIPE[3]*c


def main():
    rows = json.load(io.open(M3, encoding='utf-8'))['data']
    print('rows = %d, domain pairs = %d' % (len(rows), len({r['domain_pair'] for r in rows})))
    out = {'n_rows': len(rows)}

    for tag, fn in (('pub', risk_pub), ('pipe', risk_pipe), ('stored', None)):
        rk, gn, missing = [], [], set()
        for r in rows:
            key = ALIAS.get(r['class'], r['class'])
            if tag == 'stored':
                v = r['risk']
            else:
                if key not in T22:
                    missing.add(r['class']); continue
                v = fn(key)
            rk.append(v); gn.append(r['gain'])
        rho = float(spearmanr(rk, gn).correlation)
        per = {}
        for dp in sorted({r['domain_pair'] for r in rows}):
            sub = []
            for r in rows:
                if r['domain_pair'] != dp: continue
                key = ALIAS.get(r['class'], r['class'])
                if tag == 'stored':
                    sub.append((r['risk'], r['gain']))
                elif key in T22:
                    sub.append((fn(key), r['gain']))
            if len(sub) >= 3:
                per[dp] = round(float(spearmanr([a for a, _ in sub], [b for _, b in sub]).correlation), 4)
        n_pos = sum(1 for v in per.values() if v > 0)
        out[tag] = {'n': len(rk), 'rho': round(rho, 4),
                    'p': round(float(spearmanr(rk, gn).pvalue), 4),
                    'per_domain_pair': per, 'n_pos': n_pos, 'missing': sorted(missing)}
        print('%-6s n=%2d  rho=%+.4f  p=%.4f  正域对 %d/%d  缺类=%s'
              % (tag, len(rk), rho, float(spearmanr(rk, gn).pvalue), n_pos, len(per), sorted(missing)))

    json.dump(out, io.open('prior_unify_rho.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('written prior_unify_rho.json')


if __name__ == '__main__':
    main()
