import json,os
from collections import defaultdict
import numpy as np,torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
H=os.environ["USERPROFILE"]; D=os.path.join(H,"modeling","dota_data","DOTA_V1.5","train")
I=os.path.join(D,"images"); L=os.path.join(D,"labelTxt-v1.5","DOTA-v1.5_train")
M=os.path.join(H,".cache","huggingface","hub","models--IDEA-Research--grounding-dino-tiny","snapshots","a2bb814dd30d776dcf7e30523b00659f4f141c71")
O=os.path.join(H,"Desktop","dota_threshold_validate_results.json")
K=os.path.join(H,"Desktop","dota_fixed_validate_cache.json")
P={"ship":["ship"],"small-vehicle":["small vehicle"]}
S=[768]; T=[.10,.12,.15,.20,.25]; N=20; OVER=128; DEV="cuda"
def iou(a,b):
 x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
 z=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
 return z/(aa+bb-z) if aa+bb-z>0 else 0
def nms(bs,ss):
 q=np.argsort(ss)[::-1].tolist();r=[]
 while q:
  i=q.pop(0);r.append(i);q=[j for j in q if iou(bs[i],bs[j])<.5]
 return [bs[i] for i in r]
def samples(c):
 out=[]
 for f in sorted(os.listdir(L)):
  if not f.endswith(".txt"):continue
  ip=os.path.join(I,f[:-4]+".png")
  if not os.path.exists(ip):continue
  g=[]
  for x in open(os.path.join(L,f),encoding="utf-8"):
   p=x.split()
   if len(p)<9 or p[8]!=c:continue
   q=list(map(float,p[:8]));xs=q[0::2];ys=q[1::2];g.append([min(xs),min(ys),max(xs),max(ys)])
  if g:out.append((ip,g))
  if len(out)==N:return out
 return out
def origins(w,h,s):
 d=s-OVER;xs=list(range(0,max(1,w-s+1),d));ys=list(range(0,max(1,h-s+1),d))
 if not xs or xs[-1]+s<w:xs.append(max(0,w-s))
 if not ys or ys[-1]+s<h:ys.append(max(0,h-s))
 return [(x,y) for y in ys for x in xs]
def raw(path,prompt,size,key):
 if key in cache:return cache[key]
 im=Image.open(path).convert("RGB");bs=[];ss=[]
 for ox,oy in origins(im.width,im.height,size):
  tile=im.crop((ox,oy,min(ox+size,im.width),min(oy+size,im.height)))
  x=proc(images=tile,text=prompt,return_tensors="pt");x={k:v.to(DEV) if torch.is_tensor(v) else v for k,v in x.items()}
  with torch.no_grad():z=model(**x)
  ts=torch.tensor([tile.size[::-1]],device=DEV);r=proc.post_process_grounded_object_detection(z,threshold=min(T),target_sizes=ts)[0]
  ids=torch.argsort(r["scores"],descending=True)[:100]
  for b,s in zip(r["boxes"][ids].cpu().tolist(),r["scores"][ids].cpu().tolist()):
   bs.append([b[0]+ox,b[1]+oy,b[2]+ox,b[3]+oy]);ss.append(float(s))
 cache[key]=[bs,ss]
 with open(K,"w") as f:json.dump(cache,f)
 return bs,ss
def metric(ps,gs):
 tp=fp=fn=0
 for pp,gg in zip(ps,gs):
  used=set()
  for p in pp:
   a=[(iou(p,g),j) for j,g in enumerate(gg) if j not in used];v,j=max(a,default=(0,-1))
   if v>=.25:used.add(j);tp+=1
   else:fp+=1
  fn+=len(gg)-len(used)
 pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0
 return {"precision":pr,"recall":re,"f1":2*pr*re/(pr+re) if pr+re else 0,"tp":tp,"fp":fp,"fn":fn}
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU not found")
os.environ["HF_HUB_OFFLINE"]="1";print("GPU:",torch.cuda.get_device_name(0))
proc=AutoProcessor.from_pretrained(M);model=AutoModelForZeroShotObjectDetection.from_pretrained(M,dtype=torch.float32).eval().to(DEV)
cache=json.load(open(K)) if os.path.exists(K) else {};result={"config":{"images":N,"sizes":S,"thresholds":T},"classes":{}}
for c,prompts in P.items():
 d=samples(c);gs=[g for _,g in d];rows=[]
 for p in prompts:
  for size in S:
   rr=[raw(path,p,size,f"{c}|{p}|{size}|{os.path.basename(path)}") for path,_ in d]
   for th in T:
    ps=[]
    for bs,ss in rr:
     ids=[i for i,x in enumerate(ss) if x>=th];ps.append(nms([bs[i] for i in ids],[ss[i] for i in ids]))
    m=metric(ps,gs);row={"prompt":p,"tile":size,"threshold":th,**m};rows.append(row);print(c,row,flush=True)
 rows.sort(key=lambda x:x["f1"],reverse=True);result["classes"][c]={"best":rows[0],"all":rows};print("BEST",c,rows[0])
with open(O,"w") as f:json.dump(result,f,indent=2)
print("Saved:",O)


