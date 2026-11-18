import json,os,sys
import numpy as np,torch
from PIL import Image
import glob; H=os.environ["USERPROFILE"]; u=glob.glob(os.path.join(H,"Desktop","实验代码","unified_eval.py"))[0]; sys.path.insert(0,os.path.dirname(u))
import unified_eval as U
RAW=os.path.join(H,"Desktop","dota_fixed_validate_cache.json")
OUT=os.path.join(H,"Desktop","dota_visual_rerank_results.json")
D=os.path.join(H,"modeling","dota_data","DOTA_V1.5","train");I=os.path.join(D,"images");L=os.path.join(D,"labelTxt-v1.5","DOTA-v1.5_train")
CFG={"ship":("ship","ship"),"small-vehicle":("small vehicle","vehicle")};N=20;SIZE=768;MIN=.1;MATCH=.25
def iou(a,b):
 x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3]);z=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return z/(aa+bb-z) if aa+bb-z>0 else 0
def nms(bs,ss):
 q=np.argsort(ss)[::-1].tolist();r=[]
 while q:
  i=q.pop(0);r.append(i);q=[j for j in q if iou(bs[i],bs[j])<.5]
 return r
def data(c):
 out=[]
 for fn in sorted(os.listdir(L)):
  if not fn.endswith(".txt"):continue
  ip=os.path.join(I,fn[:-4]+".png")
  if not os.path.exists(ip):continue
  g=[]
  for x in open(os.path.join(L,fn),encoding="utf-8"):
   p=x.split()
   if len(p)<9 or p[8]!=c:continue
   q=list(map(float,p[:8]));xs=q[0::2];ys=q[1::2];g.append([min(xs),min(ys),max(xs),max(ys)])
  if g:out.append((ip,g))
  if len(out)==N:return out
 return out
def curve(rows,gts):
 rows.sort(key=lambda x:x[2],reverse=True);used=[set() for _ in gts];tp=[];fp=[]
 for im,b,s in rows:
  z=[(iou(b,g),j) for j,g in enumerate(gts[im]) if j not in used[im]];v,j=max(z,default=(0,-1));ok=v>=MATCH
  if ok:used[im].add(j)
  tp.append(ok);fp.append(not ok)
 tc=np.cumsum(tp);fc=np.cumsum(fp);ng=sum(map(len,gts));rec=tc/ng;pre=tc/(tc+fc);f=2*pre*rec/(pre+rec+1e-12);ap=0
 for t in np.linspace(0,1,11):
  m=rec>=t;ap+=(float(pre[m].max()) if m.any() else 0)/11
 k=int(np.argmax(f));return {"ap25":float(ap),"best_f1":float(f[k]),"threshold":float(rows[k][2]),"precision":float(pre[k]),"recall":float(rec[k]),"tp":int(tc[k]),"fp":int(fc[k]),"fn":int(ng-tc[k])}
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU not found")
raw=json.load(open(RAW));bank=U.get_bank();dino,pre=U.get_dinov2();ans={"config":{"images":N,"tile":SIZE,"gd_min":MIN,"iou":MATCH,"layer":"L11"},"classes":{}}
for c,(prompt,source) in CFG.items():
 ds=data(c);gts=[g for _,g in ds];pos=torch.nn.functional.normalize(bank[source][10]["prototype"].float(),dim=0)
 other=torch.stack([bank[x][10]["prototype"].float() for x in bank if x!=source]);other=torch.nn.functional.normalize(other,dim=1);neg=torch.nn.functional.normalize(other[(other@pos).topk(3).indices].mean(0),dim=0)
 rows={"gd":[],"vep":[],"cdvm":[]}
 for im,(path,g) in enumerate(ds):
  key=f"{c}|{prompt}|{SIZE}|{os.path.basename(path)}";bs,ss=raw[key];ids=[i for i,s in enumerate(ss) if s>=MIN];bs=[bs[i] for i in ids];ss=[ss[i] for i in ids];ids=nms(bs,ss);bs=[bs[i] for i in ids];ss=[ss[i] for i in ids]
  image=Image.open(path).convert("RGB");feats=torch.zeros((len(bs),768));crops=[];valid=[]
  for j,b in enumerate(bs):
   x1,y1=max(0,int(b[0])),max(0,int(b[1]));x2,y2=min(image.width,int(b[2])),min(image.height,int(b[3]))
   if x2>x1+10 and y2>y1+10:crops.append(image.crop((x1,y1,x2,y2)));valid.append(j)
  for k in range(0,len(crops),64):
   x=torch.stack([pre(z) for z in crops[k:k+64]]).to(U.DEVICE)
   with torch.no_grad():f=dino(x).hidden_states[11][:,0,:]
   f=torch.nn.functional.normalize(f,dim=1).cpu()
   for j,v in zip(valid[k:k+64],f):feats[j]=v
  gd=torch.tensor(ss);ps=feats@pos;ns=feats@neg;scores={"gd":gd,"vep":torch.sqrt(torch.clamp(gd,1e-6,1)*torch.clamp(ps,1e-6,1)),"cdvm":torch.sqrt(torch.clamp(gd,1e-6,1)*torch.clamp(ps-.5*ns,1e-6,1))}
  for name,s in scores.items():rows[name].extend((im,b,float(v)) for b,v in zip(bs,s))
  print(c,im+1,"/",len(ds),len(bs),flush=True)
 ans["classes"][c]={name:curve(x,gts) for name,x in rows.items()};print(c,ans["classes"][c],flush=True)
with open(OUT,"w") as f:json.dump(ans,f,indent=2)
print("Saved:",OUT)
