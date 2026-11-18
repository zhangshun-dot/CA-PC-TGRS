# -*- coding: utf-8 -*-
"""remove_van_samples.py — 删除van→car的样本，类名固定8个
"""
import json
from pathlib import Path

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

def main():
    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f'原始样本数: {len(all_samples)}')

    # 删除van→car的样本
    van_samples = [s for s in all_samples if s.get('src_cls') == 'van']
    print(f'van→car样本数: {len(van_samples)}')
    for s in van_samples:
        print(f'  {s["src"]}/{s["src_cls"]} -> {s["target_domain"]}/{s["target_cls"]}')

    all_samples = [s for s in all_samples if s.get('src_cls') != 'van']
    print(f'删除后样本数: {len(all_samples)}')

    # 统计类别
    classes = sorted(set(s['target_cls'] for s in all_samples))
    print(f'\n类别数: {len(classes)}')
    print(f'类别: {classes}')

    # 保存
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {samples_file}')


if __name__ == '__main__':
    main()
