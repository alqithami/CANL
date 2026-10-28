#!/usr/bin/env python3
"""Synthetic focused-confirmation contracts; never opens experiment data."""
from pathlib import Path
import argparse,copy,json,tempfile
import numpy as np
import confirmation as c
import data_core as core

def run(tensors=False):
    p=c.protocol();jobs=c.job_list(p);checks=[]
    assert c.METHODS==('ce','ratio_dcr','full_dcr','plain_fisher')
    assert len(jobs)==200 and sum(j['kind']=='treatment'for j in jobs)==160
    assert sum(j.get('probe_steps',0)for j in jobs)==0
    updates=sum(j['steps']if j['kind']=='shared_dcr_prefix'else j['treatment_steps']for j in jobs)
    assert updates==p['physical_optimizer_updates']['total']==4674150
    assert all(j['steps_per_epoch']==j['label_count']//100 for j in jobs)
    assert all(j['total_backbone_epochs']==200 and j['treatment_epochs']==(199 if j['method']in('ratio_dcr','full_dcr') else 200)for j in jobs if j['kind']=='treatment')
    checks.append('exact160treatments40prefixes200stages4674150updates')
    assert all(j['config']==c.read(c.HERE/'SELECTION.json')[j['method']]['config']for j in jobs if j['kind']=='treatment')
    assert p['official_test_access']is False and p['automatic_test_release']is False
    checks.append('selected_development_configs_unchanged_and_test_sealed')
    for count in p['allowed_shard_counts']:
        shards=[c.shard_jobs(p,i,count)for i in range(count)]
        assert set(j['id']for shard in shards for j in shard)==set(j['id']for j in jobs)
        assert sum(len(shard)for shard in shards)==200
        assert all(len({j['seed']for j in shard})==10//count for shard in shards)
        assert all(all(j['seed_index']%count==i for j in shard)for i,shard in enumerate(shards))
    checks.append('shards_partition_global_matrix_without_overlap_or_missing_pairs')
    pool=np.arange(45000,dtype=np.int64);labels=np.arange(50000)%100
    for seed in p['seeds']:
        previous=np.array([],dtype=np.int64)
        for count in p['label_counts']:
            ids=core.uniform_labeled(pool,seed,count)
            assert np.array_equal(ids,np.random.default_rng(seed).permutation(pool)[:count])
            assert np.array_equal(ids[:len(previous)],previous)and len(np.unique(ids))==count
            assert core.support(labels,ids)['status']=='SUPPORTED'
            plan=core.batches(labels,ids,seed,0,steps=count//100)
            assert plan.shape==(count//100,100)and np.isin(plan,ids).all()
            assert all(len(np.unique(row))==100 and sorted(np.unique(labels[row],return_counts=True)[1])==[10]*10 for row in plan)
            previous=ids
    checks+=['uniform_nested_label_draws_independent_of_labels','all_budgets_have_correct_paired_acquired_only_batch_plans']
    unsupported=np.concatenate([np.arange(k,50000,100)[:9 if k==0 else 10]for k in range(100)])
    assert core.support(labels,unsupported)['status']=='UNSUPPORTED_CLASS_SUPPORT'
    checks.append('unsupported_support_blocks_without_replacement_draw')
    with tempfile.TemporaryDirectory()as td:
        out=Path(td);j=next(x for x in jobs if x.get('method')=='ce');d=out/'jobs'/j['id'];d.mkdir(parents=True)
        ln,rn=c.pair_files(j)
        for n,b in [(ln,b'ids'),(rn,b'raw')]: (out/n).write_bytes(b)
        core.write(out/'FREEZE.json',{'source_binding':{'sha256':'f'*64},'files':{n:core.sha(out/n)for n in [ln,rn]}});core.write(out/'JOB_LIST.json',[j])
        (d/'CHECKPOINT.pt').write_bytes(b'checkpoint');core.write(d/'HISTORY_training.json',[{'epoch':200}])
        receipt={'job_id':j['id'],'seed':j['seed'],'label_count':j['label_count'],'pair_id':j['pair_id'],'method':j['method'],'candidate_index':j['config']['candidate_index'],'freeze_sha256':core.sha(out/'FREEZE.json'),'source_binding_sha256':'f'*64,'planned_job':j,'official_test_accessed':False,'label_ids_sha256':core.sha(out/ln),'raw_initialization_sha256':core.sha(out/rn),'checkpoint_sha256':core.sha(d/'CHECKPOINT.pt'),'training':{'history_file':'HISTORY_training.json','history_sha256':core.sha(d/'HISTORY_training.json'),'updates':j['treatment_steps'],'image_views':j['treatment_steps']*100},'status':'CONFIRMATION_TREATMENT_VERIFIED','checkpoint_reload_verification':{'passed':True},'development':{'count':5000}}
        c.verify_job(out,receipt)
        for key,value in [('seed',0),('label_count',9000),('pair_id','wrong'),('official_test_accessed',True),('planned_job',{}),('method','nonl'),('candidate_index',999),('label_ids_sha256','0'*64)]:
            bad=copy.deepcopy(receipt);bad[key]=value
            try:c.verify_job(out,bad)
            except RuntimeError:pass
            else:raise AssertionError('Receipt change accepted:'+key)
        (d/'CHECKPOINT.pt').write_bytes(b'corrupt')
        try:c.verify_job(out,receipt)
        except RuntimeError:pass
        else:raise AssertionError('Corrupt checkpoint accepted')
    checks+=['seed_budget_method_recipe_test_and_label_rebinding_refused','checkpoint_corruption_refused']
    # Every pair's DCR preparation lookup must be a legal Path and match the job.
    assert all((Path('/synthetic')/'jobs'/(j['pair_id']+'_shared_dcr_prefix')).name in {x['id']for x in jobs if x['kind']=='shared_dcr_prefix'}for j in jobs if j.get('method')in('ratio_dcr','full_dcr'))
    checks.append('DCR_prefix_lookup_bound_to_each_seed_budget_pair')
    if tensors:
        torch=core.load_torch();import methods
        core.setseed(331,torch);rawnet=core.model_new(torch);raw={'model':rawnet.state_dict()};del rawnet
        images=np.random.default_rng(9).integers(0,256,(20,3,32,32),dtype=np.uint8);yy=np.repeat(np.arange(4),5).astype(np.int64);ids=np.arange(20);target=methods.FrozenTargets(0.,.01)
        for method in c.METHODS:
            cfg=p['methods'][method]['configs'][0];core.setseed(223,torch);net=c.model_for(method,raw,cfg,torch).train();opt=torch.optim.SGD(net.parameters(),lr=cfg['learning_rate'])
            loss,_=c.step_loss(net,method,images,yy,ids,112,torch,target,cfg);assert bool(torch.isfinite(loss));loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),float('inf'),error_if_nonfinite=True);opt.step();assert c.logits_for(net,method,core.tensor(images,ids,'cpu',torch),torch).shape==(20,100)
            del net,opt,loss
        checks.append('synthetic_forward_backward_updates_all_four_frozen_method_paths')
    return {'status':'PASSED','scope':'synthetic focused confirmation contracts only; no real data, test evaluation or scientific checkpoints','torch_checks_run':tensors,'checks':checks}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--with-torch',action='store_true');p.add_argument('--report',type=Path);a=p.parse_args();r=run(a.with_torch)
    if a.report:core.write(a.report,r)
    print(json.dumps(r,indent=2))
