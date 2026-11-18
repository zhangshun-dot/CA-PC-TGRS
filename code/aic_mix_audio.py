# -*- coding: utf-8 -*-
"""aic_mix_audio.py — 给演示视频混入背景音乐
用法:
  python aic_mix_audio.py <手机传来的音频/视频文件>
音频可以是 .mp3 / .m4a / .wav，或抖音下载的 .mp4（自动抽音轨）。
输出: aic_demo_video_with_music.mp4（音乐循环铺满、音量 0.4、结尾 2s 淡出）
"""
import sys, json, subprocess
from pathlib import Path

VIDEO = Path(r'C:\Users\张三\Desktop\实验数据\aic_demo_video.mp4')
FF_DIR = (r'C:\Users\张三\AppData\Local\Microsoft\WinGet\Packages'
          r'\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe'
          r'\ffmpeg-9.0.1-full_build\bin')
FF = str(Path(FF_DIR) / 'ffmpeg.exe')
FP = str(Path(FF_DIR) / 'ffprobe.exe')
VOL = 0.4


def duration(p):
    o = subprocess.run([FP, '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'json', p], capture_output=True, text=True)
    return float(json.loads(o.stdout)['format']['duration'])


def main():
    if len(sys.argv) < 2:
        print('用法: python aic_mix_audio.py <音频或含音轨的视频文件>')
        return
    audio = sys.argv[1]
    out = VIDEO.with_name(VIDEO.stem + '_with_music.mp4')
    d = duration(str(VIDEO))
    print(f'视频时长 {d:.1f}s, 混入 {audio} ...')
    cmd = [FF, '-i', str(VIDEO), '-stream_loop', '-1', '-i', audio,
           '-map', '0:v:0', '-map', '1:a:0',
           '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest',
           '-af', f'volume={VOL},afade=t=out:st={max(0.0, d - 2):.1f}:d=2',
           str(out)]
    subprocess.run(cmd, check=True)
    print(f'完成 → {out}')


if __name__ == '__main__':
    main()
