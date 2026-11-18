# -*- coding: utf-8 -*-
"""以 DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP 启动 SigLIP 补跑,
使实验进程脱离 bash 的进程组 —— 否则 bash 工具超时(120s)给进程组发 SIGTERM
会把后台实验一起杀掉(本轮已发生两次)。

用法: python launch_siglip_detached.py
日志: 实验代码/cvlm_siglip_E.log
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'C:\Users\张三\vg\Scripts\python.exe'
SCRIPT = os.path.join(HERE, 'run_cross_vlm_cellmajor_resume.py')
LOG = os.path.join(HERE, 'cvlm_siglip_E.log')

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

env = dict(os.environ)
env.update({
    'HF_HUB_OFFLINE': '1',
    'CVLM_BS_SIGLIP': '16',       # 8GB 卡上 64 会 OOM
    'AC_SPC_DET_DEVICE': 'cuda',
    'PYTHONUNBUFFERED': '1',
})

cmd = [PY, '-u', SCRIPT, '--vlms', 'siglip_so400m', '--merge']
f = open(LOG, 'w', encoding='utf-8')
p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT,
                     creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                     close_fds=True)
print('launched pid', p.pid, '-> log', LOG)
time.sleep(5)
print('alive:', p.poll() is None)
