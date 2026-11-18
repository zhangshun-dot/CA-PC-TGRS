import zipfile, re, sys
from pathlib import Path
p = r"C:/Users/张三/Documents/xwechat_files/wxid_30izkjhlx5m822_90e5/temp/RWTemp/2026-09/98b4a19a06371e87c4770f9da386f571/A-CSPC论文汇报(2).pptx"
z = zipfile.ZipFile(p)
slides = sorted([n for n in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml$', n)], key=lambda x:int(re.findall(r'\d+',x)[0]))
for s in slides:
    xml = z.read(s).decode('utf-8', 'ignore')
    # strip tags, keep text from <a:t>
    texts = re.findall(r'<a:t>(.*?)</a:t>', xml, re.S)
    joined = ' '.join(texts)
    print(f"\n===== {s} =====")
    print(joined[:1500])
