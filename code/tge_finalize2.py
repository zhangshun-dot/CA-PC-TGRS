# -*- coding: utf-8 -*-
"""删除文末"## 图表规划"工作痕迹表。"""
p = r'C:\Users\张三\Desktop\论文初稿\acspc_中文论文初稿_新增实验整合版 - 副本.md'
s = open(p, encoding='utf-8').read()
lines = s.split('\n')
ci = next(i for i, l in enumerate(lines) if l.startswith('## 图表规划'))
removed = len(lines) - ci
lines = lines[:ci]
while lines and lines[-1].strip() == '':
    lines.pop()
open(p, 'w', encoding='utf-8').write('\n'.join(lines))
print('删除图表规划 %d 行, 剩余 %d 行' % (removed, len(lines)))
