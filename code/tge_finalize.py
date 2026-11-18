# -*- coding: utf-8 -*-
"""定稿: 备份后删除版本说明(3-13行) + 文末核对清单(509行起), 保留投稿正文与补充材料。"""
import shutil

p = r'C:\Users\张三\Desktop\论文初稿\acspc_中文论文初稿_新增实验整合版 - 副本.md'
bak = r'C:\Users\张三\Desktop\论文初稿\acspc_中文论文初稿_新增实验整合版 - 副本.工作稿备份.md'
shutil.copy(p, bak)
print('备份 →', bak)

s = open(p, encoding='utf-8').read()
lines = s.split('\n')

# 1) 删版本说明 ("> **版本说明" 到 "## 摘要" 之前)
vi = next(i for i, l in enumerate(lines) if l.startswith('> **版本说明'))
ai = next(i for i, l in enumerate(lines) if l.startswith('## 摘要'))
removed_ver = ai - vi
lines = lines[:vi] + lines[ai:]

# 2) 删核对清单 ("## 写作与投稿前核对清单" 到末尾)
ci = next(i for i, l in enumerate(lines) if l.startswith('## 写作与投稿前核对清单'))
removed_ck = len(lines) - ci
lines = lines[:ci]

# 3) 清理尾部空行
while lines and lines[-1].strip() == '':
    lines.pop()

out = '\n'.join(lines)
open(p, 'w', encoding='utf-8').write(out)
print('删除版本说明 %d 行, 核对清单 %d 行' % (removed_ver, removed_ck))
print('剩余总行数:', len(lines))
