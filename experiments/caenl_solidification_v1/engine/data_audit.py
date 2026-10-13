"""Read-only image audit and deterministic exclusions; raw images are never moved."""
import collections,hashlib,json,re,struct
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from support import sha256,atom_json,atom_npz,read_json,digest

def group_policy(records):
    groups=collections.defaultdict(list)
    for r in records:groups[r['pixel_sha256']].append(r)
    kept=[];excluded=[];duplicates=[]
    for identity,rows in sorted(groups.items()):
        ordered=sorted(rows,key=lambda r:r['path']);labels={r['class_id'] for r in rows};splits={r['split'] for r in rows}
        if len(rows)>1:
            duplicates.append(dict(pixel_sha256=identity,conflicting_labels=len(labels)>1,cross_split=len(splits)>1,members=ordered))
        if len(labels)>1:reason='conflicting_labels'
        elif len(splits)>1:reason='cross_split_content'
        else:reason=None
        if reason:excluded.extend(dict(**r,reason=reason) for r in ordered)
        else:
            kept.append(ordered[0]);excluded.extend(dict(**r,reason='same_label_duplicate') for r in ordered[1:])
    return sorted(kept,key=lambda r:r['path']),sorted(excluded,key=lambda r:r['path']),duplicates

def decoded_hash(path):
    with Image.open(path) as im:
        rgb=im.convert('RGB');return hashlib.sha256(struct.pack('>II',*rgb.size)+rgb.tobytes()).hexdigest()

def audit_catalog(original,cfg,out,progress):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    binding=digest(dict(manifest=cfg['expected_image_manifest_sha256'],policy=cfg['data_policy']))
    done=out/'DONE.json'
    if done.exists():
        d=read_json(done)
        if d['binding']!=binding:raise ValueError('Data-audit policy changed')
        for rel,h in d['payload'].items():
            if sha256(out/rel)!=h:raise ValueError('Data audit record changed: '+rel)
        retained=read_json(out/'retained.json');summary=read_json(out/'summary.json')
    else:
        records=[]
        for split in ('train','val'):
            for i,r in enumerate(original.rows[split]):
                path=original.root/r['path'];raw=sha256(path)
                if raw!=r['sha256']:raise ValueError('Dataset bytes changed: '+r['path'])
                records.append(dict(path=r['path'],split=split,original_index=i,class_id=int(r['class_id']),file_sha256=raw,pixel_sha256=decoded_hash(path)))
                if len(records)%2000==0:progress(f'Content audit: {len(records):,} images decoded and hashed')
        retained,excluded,duplicates=group_policy(records)
        # This check only applies where the standard ImageNet synset filename is present.
        # It is an internal naming check, not independent authentication of image labels.
        bad=[];checked=0
        for r in retained:
            m=re.match(r'^(n\d{8})_',Path(r['path']).name)
            if m:
                checked+=1
                if m.group(1)!=Path(r['path']).parts[1]:bad.append(r['path'])
        counts={split:np.bincount([r['class_id'] for r in retained if r['split']==split],minlength=cfg['num_classes']).tolist() for split in ('train','val')}
        summary=dict(binding=binding,original_images=len(records),retained_images=len(retained),excluded_images=len(excluded),
            counts_by_class=counts,duplicate_groups=len(duplicates),conflicting_label_groups=sum(r['conflicting_labels'] for r in duplicates),
            cross_split_groups=sum(r['cross_split'] for r in duplicates),reason_counts=dict(collections.Counter(r['reason'] for r in excluded)),
            filename_synset_checked=checked,retained_filename_synset_mismatches=bad,
            upstream_image_authenticity_verified=False,near_duplicates_certified_absent=False)
        for name,value in [('all_images.json',records),('retained.json',retained),('excluded.json',excluded),('duplicates.json',duplicates),('summary.json',summary)]:atom_json(out/name,value)
        if bad:raise ValueError('Retained filename/folder label conflicts require inspection; report saved')
        if min(counts['train'])<100 or min(counts['val'])<2:raise ValueError('Insufficient per-class support after exclusions; no training started')
        atom_json(done,dict(binding=binding,payload={name:sha256(out/name) for name in ['all_images.json','retained.json','excluded.json','duplicates.json','summary.json']}))
    rows={split:[dict(original.rows[split][r['original_index']]) for r in retained if r['split']==split] for split in ('train','val')}
    cat=SimpleNamespace(root=original.root,rows=rows,labels=np.array([r['class_id'] for r in rows['train']],dtype=np.int64),val_labels=np.array([r['class_id'] for r in rows['val']],dtype=np.int64))
    # A single identity per retained image also separates any subsequent training holdout.
    identities=[r['pixel_sha256'] for r in retained]
    if len(identities)!=len(set(identities)):raise ValueError('Content identities not unique after filtering')
    atom_npz(out/'original_indices.npz',train=np.array([r['original_index'] for r in retained if r['split']=='train']),val=np.array([r['original_index'] for r in retained if r['split']=='val']))
    return cat,summary
