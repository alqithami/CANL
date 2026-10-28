"""Train-cache-only helpers adapted from the verified finite CIFAR pilot.

This module has no archive, network or official-test dataset reader.
"""
from __future__ import annotations
import contextlib, hashlib, json, math, os, random
from pathlib import Path
import numpy as np

MEAN=(.5071,.4867,.4408)
STD=(.2675,.2565,.2761)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def objsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(t,path)

def uniform_labeled(pool,seed,count=9000):
    """Uniform identities: deliberately accepts no labels argument."""
    pool=np.asarray(pool,dtype=np.int64)
    if len(pool)!=45000 or len(np.unique(pool))!=45000 or count not in (4500,9000,18000,27000):raise ValueError('Frozen pool/budget mismatch')
    return np.random.default_rng(seed).permutation(pool)[:count]

def support(labels,labeled):
    counts=np.bincount(np.asarray(labels)[labeled],minlength=100)
    return {'status':'SUPPORTED' if len(counts)==100 and int(counts.min())>=10 else 'UNSUPPORTED_CLASS_SUPPORT','counts':counts.tolist(),'minimum':int(counts.min())}

def batches(labels,labeled,seed,epoch,stage=1,steps=90):
    rng=np.random.default_rng(np.random.SeedSequence([seed,stage,epoch,32019]))
    by_class={c:np.asarray(labeled)[np.asarray(labels)[labeled]==c] for c in range(100)}
    if any(len(x)<10 for x in by_class.values()):raise RuntimeError('UNSUPPORTED_CLASS_SUPPORT; no replacement draw allowed')
    result=[]
    for _ in range(steps):
        ids=np.concatenate([rng.choice(by_class[int(c)],10,replace=False) for c in rng.choice(100,10,replace=False)])
        result.append(ids[rng.permutation(len(ids))].astype(np.int64))
    return np.asarray(result)

def augseed(seed,epoch,step,view=0):
    return int(np.random.SeedSequence([seed,1,epoch,step,9981,view]).generate_state(1)[0])

def setseed(seed,torch):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def load_torch():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    torch.set_num_threads(min(8,os.cpu_count() or 1));torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch

def model_new(torch):
    from torchvision.models import resnet18
    net=resnet18(weights=None,num_classes=100)
    net.conv1=torch.nn.Conv2d(3,64,3,stride=1,padding=1,bias=False);net.maxpool=torch.nn.Identity()
    return net

def forward(net,x,torch):
    z=net.relu(net.bn1(net.conv1(x)));z=net.maxpool(z)
    for layer in (net.layer1,net.layer2,net.layer3,net.layer4):z=layer(z)
    z=torch.flatten(net.avgpool(z),1)
    return net.fc(z),z

def tensor(images,ids,device,torch,seed=None):
    x=torch.from_numpy(np.array(images[ids],copy=True)).float().div_(255)
    if seed is not None:
        g=torch.Generator().manual_seed(seed);offsets=torch.randint(0,9,(len(ids),2),generator=g);flip=torch.rand(len(ids),generator=g)<.5
        x=torch.nn.functional.pad(x,(4,4,4,4),mode='reflect')
        x=torch.stack([xx[:,int(a):int(a)+32,int(b):int(b)+32] for xx,(a,b) in zip(x,offsets)])
        x[flip]=x[flip].flip(-1)
    return ((x-torch.tensor(MEAN)[None,:,None,None])/torch.tensor(STD)[None,:,None,None]).to(device)

def supcon_views(images,ids,device,torch,seed):
    from PIL import Image
    from torchvision import transforms
    augment=transforms.Compose([transforms.RandomResizedCrop(32,scale=(.2,1.)),transforms.RandomHorizontalFlip(),
       transforms.RandomApply([transforms.ColorJitter(.4,.4,.4,.1)],p=.8),transforms.RandomGrayscale(p=.2),
       transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    arrays=[Image.fromarray(np.asarray(images[int(i)]).transpose(1,2,0)) for i in ids]
    setseed(seed,torch);a=torch.stack([augment(x) for x in arrays]).to(device)
    setseed(seed+1,torch);b=torch.stack([augment(x) for x in arrays]).to(device)
    return a,b

def source_view(images,ids,device,torch,seed):
    """Pinned external CIFAR single-view loader: resized crop and flip."""
    from PIL import Image
    from torchvision import transforms
    augment=transforms.Compose([transforms.RandomResizedCrop(32,scale=(.2,1.)),transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    setseed(seed,torch)
    return torch.stack([augment(Image.fromarray(np.asarray(images[int(i)]).transpose(1,2,0))) for i in ids]).to(device)

def save_checkpoint(path,x,torch):
    t=path.with_name(path.name+'.tmp');torch.save(x,t);os.replace(t,path)

def state_sha(state):
    """Stable tensor-content digest, independent of torch ZIP serialization."""
    h=hashlib.sha256()
    for k in sorted(state):
        t=state[k].detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()
