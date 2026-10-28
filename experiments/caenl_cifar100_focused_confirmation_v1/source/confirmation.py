#!/usr/bin/env python3
"""Finite focused four-method CONFIRMATION TRAINING ONLY; no official-test reader or release command.

Train-cache inputs come from the independently verified engineering pilot.
Every job is materialized before training; partial optimizer attempts block
reruns. Changing an outcome cannot change the recipe or the finite job list.
"""
from __future__ import annotations
import argparse, datetime, fcntl, hashlib, json, math, os, shutil, signal, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import data_core as core

HERE=Path(__file__).resolve().parent
METHODS=('ce','ratio_dcr','full_dcr','plain_fisher')
PILOT_BINDING='a5b69520bf0c7d462dd24f090fbf9b6eae1cd02b4b65bf90defc4297fae22e05'
INPUTS=('train_images.npy','train_labels.npy','pool_ids.npy','development_ids.npy')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def identity(pid):
    try:return {'pid':pid,'start_ticks':Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError,IndexError):return None

def protocol():
    p=read(HERE/'frozenprotocol.json');selection=read(HERE/'SELECTION.json');development=read(HERE/'DEVELOPMENT_RESULTS.json');enable=read(HERE/'CONFIRMATION_ENABLE.json')
    required={'stage':'CONFIRMATION_TRAINING_ONLY','seeds':list(range(92001,92011)),'label_counts':[4500,9000,18000,27000],'label_fractions':[.1,.2,.4,.6],'train_pool_size':45000,'development_size':5000,'batch_size':100,'classes_per_batch':10,'examples_per_class':10,'warmup_epochs':1,'anchor_batches':16,'official_test_access':False,'confirmation_enabled':True,'automatic_next_stage':False,'automatic_test_release':False}
    for k,v in required.items():
        if p.get(k)!=v:raise RuntimeError('Frozen executable/protocol mismatch: '+k)
    if set(p['methods'])!=set(METHODS) or set(selection)!=set(METHODS):raise RuntimeError('Exactly four focused methods required')
    if core.sha(HERE/'DEVELOPMENT_RESULTS.json')!=p['development_results_sha256'] or core.sha(HERE/'SELECTION.json')!=p['selection_sha256']:raise RuntimeError('Development selection source changed')
    if {m:development['selected_configurations'][m] for m in METHODS}!=selection or development['status']!='DEVELOPMENT_VERIFIED_CONFIRMATION_NOT_ENABLED' or development['official_test_accessed'] is not False:raise RuntimeError('Wrong development results')
    if enable['status']!='CONFIRMATION_TRAINING_ENABLED' or enable['official_test_access'] is not False or enable['automatic_test_release'] is not False or enable['seeds']!=p['seeds'] or enable['label_counts']!=p['label_counts'] or enable['selection_sha256']!=p['selection_sha256']:raise RuntimeError('Missing explicit matching confirmation-training enable')
    if enable['methods']!=list(METHODS) or enable['global_treatments']!=160 or enable['global_training_stages']!=200:raise RuntimeError('Explicit enable belongs to another method/count scope')
    spec=read(HERE/'ANALYSIS_SPEC.json')
    import analyze
    if spec['methods']!=list(METHODS) or spec['confirmation_seeds']!=p['seeds'] or spec['label_counts']!=p['label_counts'] or spec['analysis']!=analyze.ANALYSIS_CONTRACT or spec['locked'] is not False or spec['test_release_receipt_sha256'] is not None:raise RuntimeError('Frozen analysis/test-release contract mismatch')
    for method in METHODS:
        if p['methods'][method]['configs']!=[selection[method]['config']]:raise RuntimeError('Retuning forbidden: '+method)
        if p['methods'][method]['epochs']!=(1000 if method in ('supcon','nonl') else 200):raise RuntimeError('Frozen treatment duration changed')
    if p['protocol_amendment']['excluded_from_confirmation']!=['supcon','nonl']:raise RuntimeError('Focused-scope amendment changed')
    return p

def source_binding():
    manifest=read(HERE/'SOURCE_MANIFEST.json')
    for n,h in manifest['files'].items():
        if core.sha(HERE/n)!=h:raise RuntimeError('Changed frozen source: '+n)
    if set(manifest['files'])!={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and p.name!='SOURCE_MANIFEST.json' and '__pycache__' not in p.parts}:
        raise RuntimeError('Frozen source tree contains unbound or missing files')
    return {'manifest_sha256':core.sha(HERE/'SOURCE_MANIFEST.json'),'files':manifest['files'],'sha256':core.objsha(manifest)}

def input_binding(pilot):
    pilot=Path(pilot).resolve();m=read(pilot/'DATA_MANIFEST.json')
    if m.get('binding',{}).get('sha256')!=PILOT_BINDING or m.get('official_test_accessed') is not False or m.get('accessed_archive_members')!=['cifar-100-python/train']:raise RuntimeError('Wrong/unsealed original pilot train cache')
    files={n:core.sha(pilot/n) for n in INPUTS}
    for n,h in files.items():
        if m['files'].get(n)!=h:raise RuntimeError('Pilot input hash mismatch: '+n)
    images=np.load(pilot/'train_images.npy',mmap_mode='r');labels=np.load(pilot/'train_labels.npy',mmap_mode='r')
    pool=np.load(pilot/'pool_ids.npy',allow_pickle=False);dev=np.load(pilot/'development_ids.npy',allow_pickle=False)
    if images.shape!=(50000,3,32,32) or images.dtype!=np.uint8 or labels.shape!=(50000,) or len(pool)!=45000 or len(dev)!=5000 or len(np.unique(np.concatenate([pool,dev])))!=50000:raise RuntimeError('Train-cache shape/partition mismatch')
    if any(np.bincount(labels[dev],minlength=100)!=50) or any(np.bincount(labels[pool],minlength=100)!=450):raise RuntimeError('Frozen stratified split mismatch')
    return {'pilot_output':str(pilot),'pilot_data_manifest_sha256':core.sha(pilot/'DATA_MANIFEST.json'),'pilot_binding_sha256':PILOT_BINDING,'files':files,'official_test_accessed':False}

def job_list(p):
    result=[]
    for seed_index,seed in enumerate(p['seeds']):
        for count in p['label_counts']:
            pair=f's{seed}_n{count}';steps=count//100
            common={'seed':seed,'seed_index':seed_index,'label_count':count,'label_fraction':count/45000,'pair_id':pair,'steps_per_epoch':steps}
            result.append({**common,'id':pair+'_shared_dcr_prefix','kind':'shared_dcr_prefix','epochs':1,'steps':steps,'max_seconds':p['warmup_max_seconds']})
            for method in METHODS:
                cfg=p['methods'][method]['configs'][0];epochs=p['methods'][method]['epochs'];trained_epochs=epochs-1 if method in ('ratio_dcr','full_dcr') else epochs
                result.append({**common,'id':pair+'_'+method,'kind':'treatment','method':method,'config':cfg,'total_backbone_epochs':epochs,'treatment_epochs':trained_epochs,'backbone_steps':epochs*steps,'treatment_steps':trained_epochs*steps,'probe_steps':100*steps if method=='supcon' else 0,'max_seconds':p['supcon_job_max_seconds'] if method=='supcon' else p['regular_job_max_seconds']})
    return result

def shard_jobs(p,index,count):
    if count not in p['allowed_shard_counts'] or not 0<=index<count:raise RuntimeError('Invalid frozen shard identity')
    return [j for j in job_list(p) if j['seed_index']%count==index]

def pair_files(job):return (f'LABELS_{job["pair_id"]}.npy',f'RAW_INITIALIZATION_{job["pair_id"]}.pt')

def prepare(args):
    preparation_started=time.monotonic();p=protocol();source=source_binding();out=args.output.resolve();pilot=args.pilot_output.resolve()
    def preparation_timeout(_sig,_frame):raise TimeoutError('Frozen preparation ceiling exhausted; preserve partial preparation')
    signal.signal(signal.SIGALRM,preparation_timeout);signal.alarm(p['rawinitprep_max_seconds'])
    if out==pilot or out in pilot.parents or pilot in out.parents or out==HERE or HERE in out.parents:raise RuntimeError('Output must be separate from source and read-only input trees')
    if not (HERE/p['power_assessment_record']).is_file():raise RuntimeError('Power/variance assessment must be frozen before preparation')
    owned=shard_jobs(p,args.shard_index,args.shard_count);out.mkdir(parents=True,exist_ok=True)
    if (out/'FREEZE.json').exists():
        f=validate_freeze(out,args.pilot_output);assert_shard_args(f,args);return
    if any(out.iterdir()):raise RuntimeError('Nonempty unbound output preserved; no overwrite')
    binding=input_binding(pilot);torch=core.load_torch();labels=np.load(pilot/'train_labels.npy',mmap_mode='r');pool=np.load(pilot/'pool_ids.npy',allow_pickle=False)
    files={};supports={};ownpairs={j['pair_id'] for j in owned}
    # Every global draw/support is frozen before any method executes. Unsupported
    # pairs are preserved and block the whole global plan without replacement.
    for seed in p['seeds']:
        for count in p['label_counts']:
            pair=f's{seed}_n{count}';labeled=core.uniform_labeled(pool,seed,count);supports[pair]=core.support(labels,labeled)
            name=f'LABELS_{pair}.npy';np.save(out/name,labeled,allow_pickle=False);files[name]=core.sha(out/name)
            if pair in ownpairs:
                initialization_seed=core.augseed(seed,count,0,99);core.setseed(initialization_seed,torch);net=core.model_new(torch)
                name=f'RAW_INITIALIZATION_{pair}.pt';core.save_checkpoint(out/name,{'model':net.state_dict(),'seed':seed,'label_count':count,'initialization_seed':initialization_seed,'source_binding_sha256':source['sha256'],'state_sha256':core.state_sha(net.state_dict())},torch);files[name]=core.sha(out/name);del net
    jobs=job_list(p);core.write(out/'JOB_LIST.json',jobs);core.write(out/'SHARD_JOB_LIST.json',owned)
    analysis_template=read(HERE/'ANALYSIS_SPEC.json');analysis_template['source_binding_sha256']=source['sha256'];core.write(out/'ANALYSIS_FROZEN_TEMPLATE.json',analysis_template);files['ANALYSIS_FROZEN_TEMPLATE.json']=core.sha(out/'ANALYSIS_FROZEN_TEMPLATE.json')
    files.update({n:core.sha(out/n) for n in ['JOB_LIST.json','SHARD_JOB_LIST.json']})
    if len(jobs)!=200 or sum(j['kind']=='treatment' for j in jobs)!=160 or sum(j.get('probe_steps',0)>0 for j in jobs)!=0:raise RuntimeError('Finite global matrix count mismatch')
    if sum(j['steps'] if j['kind']=='shared_dcr_prefix' else j['treatment_steps']+j['probe_steps'] for j in jobs)!=4674150:raise RuntimeError('Global update ceiling mismatch')
    freeze={'created_utc':now(),'stage':'CONFIRMATION_TRAINING_ONLY','source_binding':source,'input_binding':binding,'files':files,'support':supports,'global_job_count':200,'global_treatment_count':160,'global_probe_count':0,'global_shared_prefix_count':40,'global_training_stage_count':200,'shard_index':args.shard_index,'shard_count':args.shard_count,'shard_job_count':len(owned),'shard_treatment_count':sum(j['kind']=='treatment' for j in owned),'shard_probe_count':sum(j.get('probe_steps',0)>0 for j in owned),'shard_shared_prefix_count':sum(j['kind']=='shared_dcr_prefix' for j in owned),'shard_training_stage_count':sum(1+(j.get('probe_steps',0)>0)for j in owned),'official_test_accessed':False,'confirmation_training_enabled':True,'test_release_enabled':False,'automatic_next_stage':False,'frozen_protocol_sha256':core.sha(HERE/'frozenprotocol.json'),'selection_sha256':core.sha(HERE/'SELECTION.json'),'power_assessment_sha256':core.sha(HERE/p['power_assessment_record']),'preparation_wall_seconds':time.monotonic()-preparation_started}
    core.write(out/'FREEZE.json',freeze)
    if any(x['status']!='SUPPORTED' for x in supports.values()):
        core.write(out/'STATUS.json',{'state':'UNSUPPORTED_CLASS_SUPPORT','supports':supports,'updated_utc':now(),'official_test_accessed':False});raise RuntimeError('Uniform draw unsupported; keep identities and block all methods; no reseed')
    print(json.dumps({'state':'CONFIRMATION_TRAINING_FROZEN','freeze_sha256':core.sha(out/'FREEZE.json'),'shard_index':args.shard_index,'shard_count':args.shard_count,'jobs':len(owned),'training_stages':freeze['shard_training_stage_count']}),flush=True)

def assert_shard_args(f,args):
    if (f['shard_index'],f['shard_count'])!=(args.shard_index,args.shard_count):raise RuntimeError('Output belongs to another frozen shard identity')

def validate_freeze(out,pilot,full_inputs=True):
    p=protocol();f=read(out/'FREEZE.json')
    if f['stage']!='CONFIRMATION_TRAINING_ONLY' or f['source_binding']!=source_binding() or f['frozen_protocol_sha256']!=core.sha(HERE/'frozenprotocol.json'):raise RuntimeError('Source/recipe/stage binding changed')
    for n,h in f['files'].items():
        if core.sha(out/n)!=h:raise RuntimeError('Prepared initialization/identity/job binding changed: '+n)
    if full_inputs and f['input_binding']!=input_binding(pilot):raise RuntimeError('Read-only input binding changed')
    if read(out/'JOB_LIST.json')!=job_list(p) or read(out/'SHARD_JOB_LIST.json')!=shard_jobs(p,f['shard_index'],f['shard_count']):raise RuntimeError('Finite global/shard job-list mismatch')
    if f['selection_sha256']!=core.sha(HERE/'SELECTION.json') or f['power_assessment_sha256']!=core.sha(HERE/p['power_assessment_record']):raise RuntimeError('Selection/assessment binding changed')
    if any(x['status']!='SUPPORTED' for x in f['support'].values()):raise RuntimeError('UNSUPPORTED_CLASS_SUPPORT')
    return f

def gpu_idle():
    x=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
    others=[int(y) for y in x.stdout.splitlines() if y.strip().isdigit() and int(y)!=os.getpid() and Path('/proc',y.strip()).exists()]
    if others:raise RuntimeError('Existing GPU processes preserved; refused overlapping run: '+str(others))

def model_for(method,raw,cfg,torch):
    import methods
    net=core.model_new(torch);net.load_state_dict(raw['model'])
    if method in ('supcon','nonl'):
        net.fc=torch.nn.Identity()
        if method=='supcon':net=methods.SupConEncoder(net,512)
        else:net=methods.NormalizedClassifier(net,512,temperature=cfg['temperature'])
    return net

def logits_for(net,method,x,torch):
    return net(x) if method in ('nonl','supcon_probe') else core.forward(net,x,torch)[0]

def evaluate(net,method,images,labels,dev,torch,guard):
    net.eval();correct=0;loss_sum=0.;device=next(net.parameters()).device
    with torch.no_grad():
        for start in range(0,len(dev),256):
            guard();ids=dev[start:start+256];x=core.tensor(images,ids,device,torch);y=torch.from_numpy(np.array(labels[ids],copy=True)).to(device)
            logits=logits_for(net,method,x,torch);loss=torch.nn.functional.cross_entropy(logits,y,reduction='sum')
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite development prediction')
            correct+=int((logits.argmax(1)==y).sum());loss_sum+=float(loss)
    return {'development_accuracy_percent':100.*correct/len(dev),'development_cross_entropy':loss_sum/len(dev),'count':len(dev),'partition':'development','selected_epoch':'final_only'}

def step_loss(net,method,images,labels,ids,seed,torch,targets,cfg):
    import methods
    device=next(net.parameters()).device
    y=torch.from_numpy(np.array(labels[ids],copy=True)).to(device);records={}
    if method=='supcon':
        a,b=core.supcon_views(images,ids,device,torch,seed)
        return methods.SupConLoss(cfg['temperature'],.07)(net.forward_views(a,b),y),records
    x=core.source_view(images,ids,device,torch,seed) if method in ('nonl','supcon_probe') else core.tensor(images,ids,device,torch,seed)
    if method=='nonl':
        _,features=net(x,return_features=True)
        return methods.NONLLoss(cfg['temperature'])(features,net.classifier.weight,y),records
    if method=='supcon_probe':return torch.nn.functional.cross_entropy(net(x),y),records
    logits,features=core.forward(net,x,torch);ce=torch.nn.functional.cross_entropy(logits,y)
    if method=='ce':return ce,{'cross_entropy':float(ce.detach()),'regularizer':0.}
    penalty,records=methods.geometry_penalty(method,features,y,targets=targets,ratio_weight=cfg['ratio_weight'],cfg=methods.GeometryConfig(floor_weight=cfg.get('floor_weight',0.)))
    return ce+penalty,{'cross_entropy':float(ce.detach()),**records}

def train_phase(net,method,job,cfg,epochs,start_epoch,images,labels,labeled,torch,targets,guard,status,jdir,phase):
    params=[p for p in net.parameters() if p.requires_grad]
    optimizer=torch.optim.SGD(params,lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    history=[];phase_start=time.monotonic();steps=0;views=0;plan_digest=hashlib.sha256();aug_digest=hashlib.sha256()
    for e in range(epochs):
        guard();absolute=e+start_epoch
        if phase=='probe':lr=cfg['learning_rate']*(cfg['decay_factor']**sum(e>=x for x in cfg['decay_epochs']))
        else:lr=cfg['learning_rate']*.5*(1+math.cos(math.pi*absolute/job['total_backbone_epochs']))
        for group in optimizer.param_groups:group['lr']=lr
        net.train();plan=core.batches(labels,labeled,job['seed'],absolute,stage=1,steps=job['steps_per_epoch']);plan_digest.update(plan.tobytes())
        sums={};epoch_start=time.monotonic()
        for step,ids in enumerate(plan):
            guard();seed=core.augseed(job['seed'],absolute,step,2 if phase=='probe' else 0)
            aug_digest.update(str(seed).encode()+b'\n');core.setseed(seed,torch)
            optimizer.zero_grad(set_to_none=True);loss,records=step_loss(net,method,images,labels,ids,seed,torch,targets,cfg)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True);optimizer.step()
            steps+=1;views+=len(ids)*(2 if method=='supcon' else 1)
            sums['loss']=sums.get('loss',0.)+float(loss.detach())
            for k,v in records.items():
                if isinstance(v,(int,float)) and math.isfinite(v):sums[k]=sums.get(k,0.)+v
            if step%30==0:status(phase=phase,epoch=e+1,epochs=epochs,step=step+1,steps=job['steps_per_epoch'],completed_updates=steps)
        record={'epoch':e+1,'absolute_backbone_epoch':absolute+1,'updates_completed':steps,'learning_rate':lr,'seconds':time.monotonic()-epoch_start,**{k:v/job['steps_per_epoch'] for k,v in sums.items()}}
        history.append(record)
        # A monotonic spend ledger, not a resume checkpoint. No automatic replay.
        with (jdir/f'SPEND_{phase}.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
        if e==0 or (e+1)%10==0 or e+1==epochs:print(json.dumps({'job':job['id'],'phase':phase,**record},allow_nan=False),flush=True)
        status(phase=phase,epoch=e+1,epochs=epochs,step=job['steps_per_epoch'],steps=job['steps_per_epoch'],completed_updates=steps)
    return {'epochs':epochs,'updates':steps,'image_views':views,'seconds':time.monotonic()-phase_start,'batch_plan_sha256':plan_digest.hexdigest(),'augmentation_seed_plan_sha256':aug_digest.hexdigest(),'history':history}

def job_worker(args):
    import methods
    p=protocol();out=args.output.resolve();freeze=validate_freeze(out,args.pilot_output);assert_shard_args(freeze,args)
    request=read(out/'RUN_REQUEST.json')
    if identity(os.getppid())!=request['coordinator_identity']:raise RuntimeError('Worker requires its live finite coordinator')
    jobs=read(out/'SHARD_JOB_LIST.json');job=next(x for x in jobs if x['id']==args.job_id);jdir=out/'jobs'/job['id'];jdir.mkdir(parents=True,exist_ok=True)
    if any(jdir.iterdir()):raise RuntimeError('Existing job attempt preserved. No silent replay or retry')
    gpu_idle();torch=core.load_torch()
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<8*1024**3:raise RuntimeError('CUDA with8GiBfree required')
    started=time.monotonic();wall_started=time.time()
    def guard():
        if time.monotonic()-started>job['max_seconds'] or time.time()>request['deadline_epoch_seconds']:raise TimeoutError('Frozen job/global time ceiling exhausted')
        if identity(os.getppid())!=request['coordinator_identity']:raise RuntimeError('Finite coordinator is no longer live; preserve worker attempt')
        if shutil.disk_usage(out).free<2*1024**3 or shutil.disk_usage(Path(os.environ['TMPDIR'])).free<2*1024**3:raise RuntimeError('Output/TMPDIRbelow2GiBfree; preserve attempt')
    def status(**x):core.write(jdir/'STATUS.json',{'state':'RUNNING','job_id':job['id'],'updated_utc':now(),'identity':identity(os.getpid()),**x})
    core.write(jdir/'ATTEMPT.json',{'started_utc':now(),'identity':identity(os.getpid()),'job':job,'freeze_sha256':core.sha(out/'FREEZE.json')})
    images=np.load(args.pilot_output/'train_images.npy',mmap_mode='r');labels=np.load(args.pilot_output/'train_labels.npy',mmap_mode='r');dev=np.load(args.pilot_output/'development_ids.npy',allow_pickle=False);labeled=np.load(out/f'LABELS_{job["pair_id"]}.npy',allow_pickle=False)
    rawpath=out/f'RAW_INITIALIZATION_{job["pair_id"]}.pt';raw=torch.load(rawpath,map_location='cpu',weights_only=False)
    targets=None;shared=None;method='ce' if job['kind']=='shared_dcr_prefix' else job['method'];cfg={'learning_rate':.1,'weight_decay':.0005,'momentum':.9} if job['kind']=='shared_dcr_prefix' else job['config']
    head_stream={'supcon':102,'nonl':103}.get(method,100);head_seed=core.augseed(job['seed'],job['label_count'],0,head_stream);core.setseed(head_seed,torch)
    net=model_for(method,raw,cfg,torch).to('cuda:0');raw_state_sha=raw['state_sha256']
    base={'job_id':job['id'],'kind':job['kind'],'seed':job['seed'],'label_count':job['label_count'],'pair_id':job['pair_id'],'freeze_sha256':core.sha(out/'FREEZE.json'),'source_binding_sha256':freeze['source_binding']['sha256'],'label_ids_sha256':core.sha(out/f'LABELS_{job["pair_id"]}.npy'),'raw_initialization_sha256':core.sha(rawpath),'raw_state_sha256':raw_state_sha,'head_initialization_stream':head_stream,'head_initialization_seed':head_seed,'official_test_accessed':False,'evaluation_partition':'development','hardware':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'precision':'float32_no_AMP_no_TF32'},'planned_job':job}
    try:
        if job['kind']=='shared_dcr_prefix':
            phasejob={**job,'total_backbone_epochs':200}
            training=train_phase(net,'ce',phasejob,cfg,1,0,images,labels,labeled,torch,None,guard,status,jdir,'shared_prefix')
            anchor=core.batches(labels,labeled,job['seed'],0,stage=2,steps=16);np.save(jdir/'ANCHOR_IDS.npy',anchor,allow_pickle=False);net.eval();features=[]
            with torch.no_grad():
                for ids in anchor:
                    guard();_,z=core.forward(net,core.tensor(images,ids,'cuda:0',torch),torch);y=torch.from_numpy(np.array(labels[ids],copy=True)).to('cuda:0');features.append((z,y))
            targets=methods.calibrate_targets(features)
            core.save_checkpoint(jdir/'CHECKPOINT.pt',{'model':net.state_dict(),'freeze_sha256':base['freeze_sha256']},torch)
            x=core.tensor(images,dev[:16],'cuda:0',torch)
            with torch.no_grad():live=core.forward(net,x,torch)[0]
            reload_net=core.model_new(torch).to('cuda:0');reload_net.load_state_dict(torch.load(jdir/'CHECKPOINT.pt',map_location='cpu',weights_only=False)['model']);reload_net.eval()
            with torch.no_grad():restored=core.forward(reload_net,x,torch)[0]
            if not torch.equal(live,restored):raise RuntimeError('Shared prefix checkpoint reload mismatch')
            prefix_reload={'passed':True,'partition':'development','count':16,'logits_sha256':hashlib.sha256(live.cpu().numpy().tobytes()).hexdigest()}
            del reload_net,x,live,restored
            core.write(jdir/'TARGETS.json',{'log_kappa_target':targets.log_kappa_target,'between_floor':targets.between_floor,'checkpoint_sha256':core.sha(jdir/'CHECKPOINT.pt'),'anchor_ids_sha256':core.sha(jdir/'ANCHOR_IDS.npy'),'calibration_batches':16})
            receipt={**base,'status':'SHARED_PREFIX_VERIFIED','training':training,'checkpoint_sha256':core.sha(jdir/'CHECKPOINT.pt'),'targets_sha256':core.sha(jdir/'TARGETS.json'),'checkpoint_reload_verification':prefix_reload,'total_seconds':time.monotonic()-started,'completed_utc':now()}
        else:
            if method in ('ratio_dcr','full_dcr'):
                shared=out/'jobs'/(job['pair_id']+'_shared_dcr_prefix');verify_job(out,read(shared/'RECEIPT.json'))
                cp=torch.load(shared/'CHECKPOINT.pt',map_location='cpu',weights_only=False);net.load_state_dict(cp['model']);t=read(shared/'TARGETS.json');targets=methods.FrozenTargets(t['log_kappa_target'],t['between_floor'])
                base['shared_prefix_checkpoint_sha256']=core.sha(shared/'CHECKPOINT.pt');base['frozen_targets_sha256']=core.sha(shared/'TARGETS.json')
            training=train_phase(net,method,job,cfg,job['treatment_epochs'],1 if shared else 0,images,labels,labeled,torch,targets,guard,status,jdir,'backbone')
            probe_training=None;encoder_sha=None
            if method=='supcon':
                # Keep complete pretraining checkpoint for encoder/probe provenance.
                core.save_checkpoint(jdir/'PRETRAIN_CHECKPOINT.pt',{'model':net.state_dict(),'freeze_sha256':base['freeze_sha256']},torch)
                encoder_sha=core.state_sha(net.encoder.state_dict());encoder=net.encoder
                base['probe_initialization_stream']=101;base['probe_initialization_seed']=core.augseed(job['seed'],job['label_count'],0,101);core.setseed(base['probe_initialization_seed'],torch);net=methods.FrozenLinearProbe(encoder,512).to('cuda:0')
                pcfg=p['methods']['supcon']['probe'];probe_training=train_phase(net,'supcon_probe',job,pcfg,pcfg['epochs'],0,images,labels,labeled,torch,None,guard,status,jdir,'probe')
                if core.state_sha(net.encoder.state_dict())!=encoder_sha:raise RuntimeError('Frozen SupCon encoder/BatchNorm changed during readout')
            scored_method='supcon_probe' if method=='supcon' else method
            ev=evaluate(net,scored_method,images,labels,dev,torch,guard)
            if any(not bool(torch.isfinite(t).all()) for t in net.state_dict().values() if t.is_floating_point()):raise FloatingPointError('Nonfinite final checkpoint')
            core.save_checkpoint(jdir/'CHECKPOINT.pt',{'model':net.state_dict(),'freeze_sha256':base['freeze_sha256'],'method':method,'config':cfg,'development':ev},torch)
            net.eval();x=core.tensor(images,dev[:16],'cuda:0',torch)
            with torch.no_grad():live=logits_for(net,scored_method,x,torch)
            loaded=torch.load(jdir/'CHECKPOINT.pt',map_location='cpu',weights_only=False)
            if method=='supcon':
                encoder=core.model_new(torch);encoder.fc=torch.nn.Identity();reloaded=methods.FrozenLinearProbe(encoder,512)
            else:reloaded=model_for(method,raw,cfg,torch)
            reloaded.load_state_dict(loaded['model']);reloaded=reloaded.to('cuda:0').eval()
            with torch.no_grad():reload_logits=logits_for(reloaded,scored_method,x,torch)
            if not torch.equal(live,reload_logits):raise RuntimeError('Checkpoint reload prediction mismatch')
            reload={'passed':True,'count':16,'partition':'development','logits_sha256':hashlib.sha256(live.cpu().numpy().tobytes()).hexdigest()}
            receipt={**base,'status':'CONFIRMATION_TREATMENT_VERIFIED','method':method,'candidate_index':cfg['candidate_index'],'training':training,'probe_training':probe_training,'development':ev,'checkpoint_sha256':core.sha(jdir/'CHECKPOINT.pt'),'checkpoint_reload_verification':reload,'total_seconds':time.monotonic()-started,'completed_utc':now()}
            receipt['attributed_shared_prefix_seconds']=read(shared/'RECEIPT.json')['total_seconds'] if shared else 0.
            receipt['total_treatment_seconds_including_shared_prefix']=receipt['total_seconds']+receipt['attributed_shared_prefix_seconds']
            receipt['attributed_backbone_updates']=job['backbone_steps']
            if method=='supcon':receipt.update({'pretrain_checkpoint_sha256':core.sha(jdir/'PRETRAIN_CHECKPOINT.pt'),'frozen_encoder_state_sha256':encoder_sha})
        # Retain detailed histories as separately bound records, keeping receipt compact.
        for key in ('training','probe_training'):
            if receipt.get(key):
                hist=receipt[key].pop('history');name='HISTORY_'+key+'.json';core.write(jdir/name,hist);receipt[key]['history_file']=name;receipt[key]['history_sha256']=core.sha(jdir/name)
        guard();core.write(jdir/'RECEIPT.json',receipt);verify_job(out,receipt)
        core.write(jdir/'STATUS.json',{'state':receipt['status'],'updated_utc':now(),'job_id':job['id'],'receipt_sha256':core.sha(jdir/'RECEIPT.json')})
        print(json.dumps({'job':job['id'],'status':receipt['status'],'total_seconds':receipt['total_seconds']}),flush=True)
    except BaseException as e:
        core.write(jdir/'STATUS.json',{'state':'BLOCKED','error':repr(e),'updated_utc':now(),'job_id':job['id'],'attempt_elapsed_seconds':time.time()-wall_started});raise

def verify_job(out,r):
    jdir=out/'jobs'/r['job_id'];f=read(out/'FREEZE.json');jobs=read(out/'JOB_LIST.json');j=next(x for x in jobs if x['id']==r['job_id'])
    if r['seed']!=j['seed'] or r['label_count']!=j['label_count'] or r['pair_id']!=j['pair_id']:raise RuntimeError('Job seed/budget pair rebinding refused')
    if r['freeze_sha256']!=core.sha(out/'FREEZE.json') or r['source_binding_sha256']!=f['source_binding']['sha256'] or r['planned_job']!=j or r['official_test_accessed'] is not False:raise RuntimeError('Job receipt/source/test binding mismatch')
    for name,key in [(f'LABELS_{j["pair_id"]}.npy','label_ids_sha256'),(f'RAW_INITIALIZATION_{j["pair_id"]}.pt','raw_initialization_sha256')]:
        if r[key]!=f['files'][name] or core.sha(out/name)!=r[key]:raise RuntimeError('Job acquired identities/raw initialization changed')
    for name,key in [('CHECKPOINT.pt','checkpoint_sha256'),('TARGETS.json','targets_sha256'),('PRETRAIN_CHECKPOINT.pt','pretrain_checkpoint_sha256')]:
        if key in r and core.sha(jdir/name)!=r[key]:raise RuntimeError('Job checkpoint/target changed')
    for key in ('training','probe_training'):
        if r.get(key) and core.sha(jdir/r[key]['history_file'])!=r[key]['history_sha256']:raise RuntimeError('Job history changed')
    expected=j['steps_per_epoch'] if j['kind']=='shared_dcr_prefix' else j['treatment_steps']
    if r['training']['updates']!=expected or r['training']['image_views']!=expected*100*(2 if j.get('method')=='supcon' else 1):raise RuntimeError('Incomplete or extra finite training updates/views')
    if j['kind']=='treatment':
        if r['status']!='CONFIRMATION_TREATMENT_VERIFIED' or not r['checkpoint_reload_verification']['passed'] or r['development']['count']!=5000 or r['method']!=j['method'] or r['candidate_index']!=j['config']['candidate_index']:raise RuntimeError('Confirmation completion/reload/method contract mismatch')
        if j['method']=='supcon' and (not r['probe_training'] or r['probe_training']['updates']!=j['probe_steps'] or r['probe_training']['image_views']!=j['probe_steps']*100):raise RuntimeError('SupCon full readout incomplete')
        if j['method'] in ('ratio_dcr','full_dcr'):
            shared=out/'jobs'/(j['pair_id']+'_shared_dcr_prefix')
            if core.sha(shared/'CHECKPOINT.pt')!=r['shared_prefix_checkpoint_sha256'] or core.sha(shared/'TARGETS.json')!=r['frozen_targets_sha256']:raise RuntimeError('Shared DCR warmup/target binding changed')
    elif r['status']!='SHARED_PREFIX_VERIFIED' or not r['checkpoint_reload_verification']['passed']:raise RuntimeError('Shared prefix not verified/reloaded')
    return r

def package_confirmation(out,pilot):
    f=validate_freeze(out,pilot);jobs=read(out/'SHARD_JOB_LIST.json');receipts=[verify_job(out,read(out/'jobs'/j['id']/'RECEIPT.json')) for j in jobs]
    treatments=[r for r in receipts if r['kind']=='treatment']
    if len(treatments)!=f['shard_treatment_count']:raise RuntimeError('Incomplete shard family')
    result={'status':'CONFIRMATION_TRAINING_VERIFIED_TEST_NOT_RELEASED','created_utc':now(),'freeze_sha256':core.sha(out/'FREEZE.json'),'source_binding_sha256':f['source_binding']['sha256'],'selection_sha256':f['selection_sha256'],'shard_index':f['shard_index'],'shard_count':f['shard_count'],'global_treatments':160,'global_training_stages':200,'completed_treatments':len(treatments),'completed_probe_fits':sum(r.get('method')=='supcon' for r in treatments),'completed_shared_prefixes':sum(r['kind']=='shared_dcr_prefix'for r in receipts),'total_training_stages':sum(1+(r.get('probe_training')is not None)for r in receipts),'actual_optimizer_updates':sum(r['training']['updates']+(r.get('probe_training')or{}).get('updates',0)for r in receipts),'actual_image_views':sum(r['training']['image_views']+(r.get('probe_training')or{}).get('image_views',0)for r in receipts),'total_job_seconds':sum(r['total_seconds']for r in receipts),'official_test_accessed':False,'test_release_enabled':False,'automatic_next_stage':False,'scientific_inference':'Training completion only. No official-test result, AULC, superiority, annotation savings, equivalence or mechanism claim. Development diagnostics are not used for reselection.','next_required':'Verify all global shards and model checkpoints, then separately authorize and freeze official-test release and analysis.','final_development_diagnostics':{r['job_id']:r['development']for r in treatments},'receipt_hashes':{r['job_id']:core.sha(out/'jobs'/r['job_id']/'RECEIPT.json')for r in receipts}}
    result['preparation_wall_seconds']=f['preparation_wall_seconds'];result['pipeline_elapsed_before_packaging_seconds']=time.time()-read(out/'RUN_REQUEST.json')['started_epoch_seconds']+f['preparation_wall_seconds']
    core.write(out/'RESULTS.json',result);payloads={};review=out/f'CAENL_CIFAR100_CONFIRMATION_SHARD{f["shard_index"]}_REVIEW.zip'
    with zipfile.ZipFile(review,'x',zipfile.ZIP_DEFLATED)as z:
        for n in ['FREEZE.json','JOB_LIST.json','SHARD_JOB_LIST.json','ANALYSIS_FROZEN_TEMPLATE.json','RESULTS.json','RUN_REQUEST.json']:
            payloads[n]=core.sha(out/n);z.write(out/n,n)
        for p in sorted((out/'jobs').rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl'):
                n=str(p.relative_to(out));payloads[n]=core.sha(p);z.write(p,n)
        for p in sorted(HERE.rglob('*')):
            if p.is_file()and '__pycache__'not in p.parts:
                n='source/'+str(p.relative_to(HERE));payloads[n]=core.sha(p);z.write(p,n)
        z.writestr('PAYLOAD_SHA256.json',json.dumps(payloads,indent=2)+'\n')
    with zipfile.ZipFile(review)as z:
        if z.testzip()is not None or any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in payloads.items()):raise RuntimeError('Review CRC/payload mismatch')
    (out/(review.name+'.sha256')).write_text(core.sha(review)+'  '+review.name+'\n');result['review_filename']=review.name
    return result

def status(out):
    value=read(out/'STATUS.json') if (out/'STATUS.json').exists() else {'state':'NOT_STARTED'}
    pid=read(out/'PID.json') if (out/'PID.json').exists() else {}
    value['coordinator_live_verified']=bool(pid.get('identity') and identity(pid['identity']['pid'])==pid['identity'])
    value['completed_treatment_receipts']=sum(1 for p in (out/'jobs').glob('*/RECEIPT.json') if read(p).get('kind')=='treatment') if (out/'jobs').exists() else 0
    if value.get('current_job'):
        j=out/'jobs'/value['current_job']/'STATUS.json';value['current_job_progress']=read(j) if j.exists() else None
    return value

def coordinator(args):
    out=args.output.resolve();p=protocol();freeze=validate_freeze(out,args.pilot_output);assert_shard_args(freeze,args)
    with (out/'RUN.lock').open('a+') as lock,(Path('/mnt/caenl/active')/'CAENL_GPU_CONFIRMATION.lock').open('a+') as gpu_lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'RUN_REQUEST.json').exists():raise RuntimeError('Existing permanent confirmation attempt preserved; no restart')
        tmp=Path(os.environ.get('TMPDIR',''))
        if not os.environ.get('TMPDIR') or not tmp.is_dir() or shutil.disk_usage(tmp).free<8*1024**3 or shutil.disk_usage(out).free<8*1024**3:raise RuntimeError('Set persistent spacious TMPDIR; both volumes need8GiBfree')
        gpu_idle();start=time.time();request={'started_utc':now(),'started_epoch_seconds':start,'deadline_epoch_seconds':start+p['overall_max_seconds'],'coordinator_identity':identity(os.getpid()),'freeze_sha256':core.sha(out/'FREEZE.json'),'finite_jobs':freeze['shard_job_count'],'training_stages':freeze['shard_training_stage_count'],'shard_index':freeze['shard_index'],'shard_count':freeze['shard_count'],'official_test_accessed':False,'automatic_next_stage':False}
        core.write(out/'RUN_REQUEST.json',request);core.write(out/'PID.json',{'identity':identity(os.getpid()),'created_utc':now()})
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        try:
            for index,j in enumerate(read(out/'SHARD_JOB_LIST.json')):
                validate_freeze(out,args.pilot_output,full_inputs=False);gpu_idle()
                remaining=request['deadline_epoch_seconds']-time.time()
                if remaining<=0:raise TimeoutError('Frozen finite campaign allowance exhausted')
                logfile=out/('LOG_'+j['id']+'.txt')
                with logfile.open('xb')as log:
                    child=subprocess.Popen([sys.executable,'-B',str(HERE/'confirmation.py'),'job','--output',str(out),'--pilot-output',str(args.pilot_output),'--job-id',j['id'],'--shard-index',str(freeze['shard_index']),'--shard-count',str(freeze['shard_count'])],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                    core.write(out/'STATUS.json',{'state':'RUNNING_CONFIRMATION','current_job':j['id'],'job_index':index+1,'job_count':freeze['shard_job_count'],'shard_index':freeze['shard_index'],'shard_count':freeze['shard_count'],'updated_utc':now(),'worker_identity':identity(child.pid),'coordinator_identity':identity(os.getpid()),'official_test_accessed':False,'automatic_next_stage':False})
                    try:code=child.wait(timeout=min(j['max_seconds'],remaining))
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        raise TimeoutError('Frozen job/global time limit reached: '+j['id'])
                    if code:raise RuntimeError('Job failed and is preserved; no retry: '+j['id']+' exit'+str(code))
                verify_job(out,read(out/'jobs'/j['id']/'RECEIPT.json'))
            result=package_confirmation(out,args.pilot_output)
            core.write(out/'STATUS.json',{'state':result['status'],'updated_utc':now(),'completed_treatment_receipts':result['completed_treatments'],'shard_index':freeze['shard_index'],'shard_count':freeze['shard_count'],'automatic_next_stage':False,'official_test_accessed':False,'review_sha256':core.sha(out/result['review_filename'])})
        except BaseException as e:
            old=read(out/'STATUS.json') if (out/'STATUS.json').exists() else {}
            core.write(out/'STATUS.json',{**old,'state':'BLOCKED','error':repr(e),'updated_utc':now(),'automatic_next_stage':False});raise

def launch(args):
    out=args.output.resolve();freeze=validate_freeze(out,args.pilot_output);assert_shard_args(freeze,args)
    with (out/'LAUNCH.lock').open('a+')as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'LAUNCH_RESERVATION.json').exists():print(json.dumps(status(out),indent=2));return
        gpu_idle()
        with (out/'LAUNCH_RESERVATION.json').open('x')as f:json.dump({'created_utc':now(),'freeze_sha256':core.sha(out/'FREEZE.json')},f)
        with (out/'COORDINATOR.log').open('ab')as log:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/'confirmation.py'),'coordinator','--output',str(out),'--pilot-output',str(args.pilot_output),'--shard-index',str(freeze['shard_index']),'--shard-count',str(freeze['shard_count'])],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
        print(json.dumps({'state':'COORDINATOR_STARTED_NOT_YET_WORKER_VERIFIED','identity':identity(child.pid),'output':str(out),'freeze_sha256':core.sha(out/'FREEZE.json')},indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','launch','coordinator','job','verify','status']);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pilot-output',type=Path,default=Path('/mnt/caenl/active/caenl-cifar100-v1/pilot_output'));parser.add_argument('--job-id');parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='launch':launch(args)
    elif args.command=='coordinator':coordinator(args)
    elif args.command=='job':job_worker(args)
    elif args.command=='status':print(json.dumps(status(args.output),indent=2))
    else:
        validate_freeze(args.output,args.pilot_output);rs=[verify_job(args.output,read(args.output/'jobs'/j['id']/'RECEIPT.json')) for j in read(args.output/'SHARD_JOB_LIST.json')];print(json.dumps({'status':'ALL_CONFIRMATION_SHARD_RECEIPTS_VERIFIED','count':len(rs),'official_test_accessed':False}))

if __name__=='__main__':main()
