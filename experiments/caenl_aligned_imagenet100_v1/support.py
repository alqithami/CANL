"""Local-only data and atomic provenance. No remote downloads or destructive cleanup."""
from __future__ import annotations
import csv, hashlib, json, math, os, random, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from PIL import Image


def utc(): return datetime.now(timezone.utc).isoformat()
def read_json(p): return json.loads(Path(p).read_text())
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def digest(o): return hashlib.sha256(json.dumps(o,sort_keys=True,allow_nan=False).encode()).hexdigest()
def seed_for(*v): return int(hashlib.sha256('|'.join(map(str,v)).encode()).hexdigest()[:8],16)%(2**31-1)
def atom_json(p,o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(o,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)
def atom_torch(p,o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f: torch.save(o,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)
def atom_npz(p,**o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**o); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)
def write_csv(p,rows):
    rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for row in rows for k in row));tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    os.replace(tmp,p)
def append_json(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f: f.write(json.dumps(o,allow_nan=False)+'\n')
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'].cpu())
    if r['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()
def amp(device):
    return torch.autocast('cuda' if str(device).startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=str(device).startswith('cuda'))
def to_device(batch,device):
    x,y,ids=batch
    return x.to(device,non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last),y.to(device),ids

def stratified_sample(ids,labels,n,seed):
    ids=np.sort(np.asarray(ids,dtype=np.int64));cs,counts=np.unique(labels[ids],return_counts=True)
    if len(np.unique(ids))!=len(ids) or not 0<n<len(ids):raise ValueError('Invalid stratified sample')
    raw=counts.astype(float)*n/len(ids);q=np.floor(raw).astype(int)
    order=np.argsort(-(raw-q),kind='stable');q[order[:n-int(q.sum())]]+=1
    rng=np.random.default_rng(seed);chosen=[]
    for c,k in zip(cs,q):chosen.extend(rng.choice(ids[labels[ids]==c],int(k),replace=False).tolist())
    out=np.sort(np.asarray(chosen,dtype=np.int64))
    if len(out)!=n or len(np.unique(out))!=n:raise RuntimeError('Stratified count mismatch')
    return out

class Catalog:
    def __init__(self,cfg):
        self.root=Path(cfg['dataset_root']); ready=read_json(self.root/'IMAGENET100_READY.json')
        if ready.get('status')!='READY':raise ValueError('ImageNet-100 not READY')
        for k,expected in [('train_images',cfg['expected_train_count']),('validation_images',cfg['expected_val_count']),('train_classes',cfg['num_classes']),('validation_classes',cfg['num_classes'])]:
            if ready.get(k)!=expected:raise ValueError('Dataset metadata mismatch: '+k)
        if sha256(self.root/'images_manifest.jsonl')!=cfg['expected_image_manifest_sha256']:raise ValueError('Manifest digest mismatch')
        if ready['class_list_sha256']!=cfg['expected_class_list_sha256']:raise ValueError('Class list digest mismatch')
        with (self.root/'class_counts.csv').open() as f:cr=sorted(csv.DictReader(f),key=lambda r:int(r['class_id']))
        self.classes=[r['wnid'] for r in cr]
        if [int(r['class_id']) for r in cr]!=list(range(cfg['num_classes'])):raise ValueError('Invalid class mapping')
        self.rows={'train':[],'val':[]}
        for line in (self.root/'images_manifest.jsonl').open():
            r=json.loads(line);p=Path(r['path']);c=int(r['class_id'])
            if p.is_absolute() or '..' in p.parts or len(p.parts)!=3 or p.parts[0] not in self.rows:raise ValueError('Unsafe image path')
            if not 0<=c<cfg['num_classes'] or p.parts[1]!=self.classes[c]:raise ValueError('Label mapping mismatch')
            self.rows[p.parts[0]].append(r)
        for split,expected in [('train',cfg['expected_train_count']),('val',cfg['expected_val_count'])]:
            self.rows[split].sort(key=lambda r:r['path'])
            if len(self.rows[split])!=expected or len({r['path'] for r in self.rows[split]})!=expected:raise ValueError('Manifest count/duplicates')
        self.labels=np.asarray([r['class_id'] for r in self.rows['train']],dtype=np.int64)
        self.val_labels=np.asarray([r['class_id'] for r in self.rows['val']],dtype=np.int64)
        if not np.all(np.bincount(self.val_labels,minlength=cfg['num_classes'])==50):raise ValueError('Validation class counts must be 50 each')
    def verify_images(self,progress):
        checked=0
        for split in ('train','val'):
            for r in self.rows[split]:
                p=self.root/r['path']
                if p.is_symlink() or not p.is_file() or sha256(p)!=r['sha256']:raise ValueError(f'Image integrity mismatch: {p}')
                checked+=1
                if checked%5000==0:progress(f'Preflight: verified {checked:,} image hashes')
        return checked

class Images(Dataset):
    def __init__(self,cat,split,seed,augmented):
        self.root=cat.root;self.rows=cat.rows[split];self.seed=seed;self.augmented=augmented
        self.aug=T.Compose([T.RandomResizedCrop(224,scale=(.08,1),ratio=(.75,4/3),interpolation=T.InterpolationMode.BILINEAR),T.RandomHorizontalFlip(.5),T.PILToTensor()])
        self.ev=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BILINEAR),T.CenterCrop(224),T.PILToTensor()])
    def __len__(self):return len(self.rows)
    def __getitem__(self,key):
        step,idx=map(int,key);r=self.rows[idx]
        with Image.open(self.root/r['path']) as im:
            im=im.convert('RGB')
            if self.augmented:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed_for('aligned_al_crop',self.seed,step,idx));x=self.aug(im)
            else:x=self.ev(im)
        return x,int(r['class_id']),idx

class BalancedBatches:
    """A step-addressable sampler: resume does not replay/advance a global RNG."""
    def __init__(self,ids,labels,seed,total_steps,classes=16,per_class=8,start_step=0):
        ids=np.sort(np.asarray(ids,dtype=np.int64));self.pools={int(c):ids[labels[ids]==c] for c in np.unique(labels[ids])}
        self.keys=np.array(sorted(self.pools));self.seed=seed;self.total=int(total_steps);self.start=int(start_step)
        self.classes=int(classes);self.k=int(per_class)
        if len(self.keys)<classes or min(map(len,self.pools.values()))<per_class:raise ValueError('Insufficient labeled class support')
    def __len__(self):return self.total-self.start
    def batch(self,step):
        rng=np.random.default_rng(seed_for('balanced_al_batch',self.seed,int(step)))
        cs=rng.choice(self.keys,self.classes,replace=False)
        ids=np.concatenate([rng.choice(self.pools[int(c)],self.k,replace=False) for c in cs]);rng.shuffle(ids)
        return [(int(step),int(i)) for i in ids]
    def __iter__(self):
        for step in range(self.start+1,self.total+1):yield self.batch(step)

def loader(cat,split,batches,seed,workers,augmented):
    kw=dict(batch_sampler=batches,num_workers=workers,pin_memory=torch.cuda.is_available(),generator=torch.Generator().manual_seed(seed_for('data_workers',seed)))
    if workers:kw.update(multiprocessing_context='spawn',prefetch_factor=2,timeout=240)
    return DataLoader(Images(cat,split,seed,augmented),**kw)
def evaluation_batches(ids,bs):return [[(0,int(i)) for i in ids[j:j+bs]] for j in range(0,len(ids),bs)]

class Backbone(nn.Module):
    def __init__(self,classes=100):super().__init__();self.net=models.resnet50(weights=None,num_classes=classes)
    def forward_features(self,x):
        n=self.net;x=n.maxpool(n.relu(n.bn1(n.conv1(x))));x=n.layer1(x);x=n.layer2(x)
        x=n.layer3(x);z3=x.mean((2,3));x=n.layer4(x);z4=x.mean((2,3))
        return n.fc(z4),{'layer3':z3,'layer4':z4}
class Model(nn.Module):
    def __init__(self,classes=100):
        super().__init__();self.model=Backbone(classes)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward_features(self,x):return self.model.forward_features((x.float()-self.mean)/self.std)
    def forward(self,x):return self.forward_features(x)[0]
def train_mode(model,initial):
    model.train()
    if not initial:
        for m in model.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
def optimizer_for(model,cfg):
    one=[];multi=[]
    for p in model.parameters():(one if p.ndim<=1 else multi).append(p)
    return torch.optim.SGD([{'params':one,'weight_decay':0.},{'params':multi,'weight_decay':cfg['weight_decay']}],lr=cfg['lr_round'],momentum=cfg['momentum'],nesterov=cfg['nesterov'])
def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
