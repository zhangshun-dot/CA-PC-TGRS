# -*- coding: utf-8 -*-
"""脱离进程组启动 v3 coarse_medium 补跑 (剩余 4/18 格)。
原因: Bash 工具 120s 超时会给整个进程组发 SIGTERM, 把同组后台实验一起杀掉。
用 DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP 起, 并显式设 Bash timeout 来等待。

GPU 上**不得**有其它任务并行(8GB 卡, 并行会 OOM 且慢 10 倍)。
"""
import os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'C:\Users\张三\vg\Scripts\python.exe'
SCRIPT = os.path.join(HERE, 'run_coarse_medium.py')
LOG = os.path.join(r'C:\Users\张三\Desktop\实验数据', 'coarse_medium_v3b.log')

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

env = dict(os.environ)
env.update({'HF_HUB_OFFLINE': '1', 'AC_SPC_DET_DEVICE': 'cuda', 'PYTHONUNBUFFERED': '1'})

cmd = [PY, '-u', SCRIPT, '--k', '60', '--media', 'sketch', 'cartoon', '--dets', 'gdino']
f = open(LOG, 'w', encoding='utf-8')
p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT,
                     creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True)
print('launched pid', p.pid, '-> log', LOG)
time.sleep(5)
print('alive:', p.poll() is None)
