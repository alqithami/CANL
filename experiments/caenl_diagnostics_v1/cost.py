"""Finite, transient deployment-cost diagnostic; no fitted model is retained.

The cached-batch experiment measures local training and feedback overhead. It
does not estimate full-job duration, accuracy, tuning efficiency, or deployment
throughput. Its inputs are read only; the completed campaign must be gated by
the launcher before ``run`` is called.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path


CONDITIONS = tuple(f"{implementation}_{treatment}"
                   for implementation in ("streamlined", "instrumented")
                   for treatment in ("entropy", "fixed", "lite"))


def _required_scatter(z, labels, cfg):
    """Original balanced scatter operations, omitting unused total/kappa outputs."""
    import torch
    import torch.nn.functional as F
    z = z if z.dtype == torch.float64 else z.float()
    h = F.normalize(z, p=2, dim=1, eps=cfg['feature_norm_epsilon'])
    classes, inv, counts = torch.unique(labels, sorted=True, return_inverse=True,
                                        return_counts=True)
    if classes.numel() < 2 or int(counts.min()) < 2:
        raise ValueError('Balanced geometry requires at least two classes/examples')
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError('Geometry loss requires balanced batches')
    incidence = F.one_hot(inv, classes.numel()).to(h.dtype)
    means = incidence.T @ h / counts.to(h.dtype)[:, None]
    weights = torch.ones_like(counts, dtype=h.dtype) / len(counts)
    grand = (weights[:, None] * means).sum(0)
    residual = (h - means[inv]).square().sum(1)
    per_class_w = incidence.T @ residual / counts.to(h.dtype)
    within = (weights * per_class_w).sum()
    between = (weights * (means - grand).square().sum(1)).sum()
    q = torch.log(within + cfg['scatter_epsilon']) - torch.log(between + cfg['scatter_epsilon'])
    return {'within': within, 'between': between, 'log_kappa': q}


def aligned_tensors(features, labels, targets, weights, cfg, instrumented=False):
    """Exact DCR scalar loss; Python monitoring extraction happens separately."""
    from aligned import class_scatter, positive_huber
    import torch
    first = next(iter(features.values()))
    loss = first.new_zeros((), dtype=torch.float32)
    stats = {}
    for layer, z in features.items():
        s = (class_scatter(z, labels, eps=cfg['scatter_epsilon'],
                           norm_eps=cfg['feature_norm_epsilon'])
             if instrumented else _required_scatter(z, labels, cfg))
        gap = s['log_kappa'] - float(targets[layer]['log_kappa_target'])
        spread_gap = math.log(float(targets[layer]['between_floor'])) - torch.log(
            s['between'] + cfg['scatter_epsilon'])
        ratio_penalty = positive_huber(gap, cfg['huber_delta'])
        spread_penalty = positive_huber(spread_gap, cfg['huber_delta'])
        weight = float(weights[layer])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Invalid regularization coefficient')
        loss = loss + weight * ratio_penalty + cfg['floor_weight'] * spread_penalty
        stats[layer] = dict(s, log_gap=gap, ratio_penalty=ratio_penalty,
                            spread_penalty=spread_penalty,
                            **({'lambda': gap.new_tensor(weight)} if instrumented else {}))
    return loss, stats


def monitoring_records(stats, instrumented):
    keys = None if instrumented else ('log_kappa', 'between')
    return {layer: {key: float(value.detach()) for key, value in s.items()
                    if keys is None or key in keys}
            for layer, s in stats.items()}


def condition_order(repetition):
    """Order is fixed without looking at timing; odd repetitions reverse it."""
    order = list(CONDITIONS)
    random.Random(90317 + repetition // 2).shuffle(order)
    return order if repetition % 2 == 0 else list(reversed(order))


def _tensor_hash(tensors):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _cached_feedback(model, cached, cfg, device):
    """Faithful original fp32 holdout CE (Lite does not use CE in its formula)."""
    import torch
    import torch.nn.functional as F
    model.eval()
    x, y = cached
    total = 0.
    with torch.no_grad():
        for start in range(0, len(y), cfg['eval_batch']):
            xx = x[start:start + cfg['eval_batch']].to(device).float().div_(255)
            xx = xx.contiguous(memory_format=torch.channels_last)
            yy = y[start:start + cfg['eval_batch']].to(device)
            total += float(F.cross_entropy(model(xx).float(), yy, reduction='sum'))
    ce = total / len(y)
    if not math.isfinite(ce):
        raise FloatingPointError('Nonfinite feedback loss')
    return ce


def _make_control(condition, targets, cfg, device, windows):
    from control import Control
    implementation, treatment = condition.split('_', 1)
    if treatment == 'lite':
        return Control('aligned_lite', targets, 62001, cfg, device, windows)
    if implementation == 'instrumented':
        return Control('none' if treatment == 'entropy' else 'aligned_fixed',
                       targets, 62001, cfg, device, windows)
    return None


def execute(model, optimizer, control, condition, cached_batches, feedback,
            targets, cfg, device, steps, progress=None):
    """One bounded segment, including initial and terminal feedback when used."""
    import torch
    import torch.nn.functional as F
    from aligned import class_scatter
    from support import amp, sync, to_device, train_mode
    implementation, treatment = condition.split('_', 1)
    instrumented = implementation == 'instrumented'
    compute_s = monitoring_s = transfer_s = 0.
    trace = []
    batch_chain = hashlib.sha256(b'BATCH_CHAIN_V1').hexdigest()
    sync(device)
    begin = time.perf_counter()
    if control is not None:
        t = time.perf_counter()
        control.begin_phase(_cached_feedback(model, feedback, cfg, device), 0)
        sync(device)
        monitoring_s += time.perf_counter() - t
    train_mode(model, False)
    for step in range(1, steps + 1):
        sync(device)
        t = time.perf_counter()
        x, y, ids = to_device(cached_batches[(step - 1) % len(cached_batches)], device)
        sync(device)
        transfer_s += time.perf_counter() - t
        optimizer.zero_grad(set_to_none=True)
        t = time.perf_counter()
        with amp(device):
            logits, features = model.forward_features(x)
            ce = F.cross_entropy(logits.float(), y)
        stats = {}
        if treatment != 'entropy':
            weights = (control.weights if control is not None else
                       {layer: cfg['lambda_fixed'] for layer in cfg['layers']})
            reg, stats = aligned_tensors(features, y, targets, weights,
                                        cfg['alignment'], instrumented)
        else:
            reg = ce.new_zeros(())
        loss = ce + reg
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite microbenchmark objective')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
            cfg['clip_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        sync(device)
        compute_s += time.perf_counter() - t
        t = time.perf_counter()
        if instrumented and treatment == 'entropy':
            with torch.no_grad():
                stats = {layer: class_scatter(z.detach(), y,
                         eps=cfg['alignment']['scatter_epsilon'],
                         norm_eps=cfg['alignment']['feature_norm_epsilon'])
                         for layer, z in features.items()}
        if control is not None:
            records = monitoring_records(stats, instrumented)
            control.observe(records)
        if instrumented:
            # Match the research path's per-step scalar monitoring. Disk writes,
            # acquisition, validation export and checkpoint serialization are excluded.
            scalars = dict(train_ce=float(ce.detach()), regularizer=float(reg.detach()),
                           preclip_grad_norm=float(grad_norm),
                           train_accuracy=float((logits.detach().argmax(1) == y).float().mean()))
            if not all(math.isfinite(v) for v in scalars.values()):
                raise FloatingPointError('Nonfinite monitoring scalar')
            scalars.update(step=step, lr=optimizer.param_groups[0]['lr'])
            for layer in cfg['layers']:
                scalars.update({f'{layer}_log_kappa': records[layer]['log_kappa'],
                                f'{layer}_between': records[layer]['between'],
                                f'{layer}_lambda': control.weights[layer]})
            trace.append(scalars)
            batch_chain = hashlib.sha256(bytes.fromhex(batch_chain) +
                ids.detach().cpu().numpy().astype('int64', copy=False).tobytes()).hexdigest()
        if control is not None and (step % cfg['control_interval'] == 0 or step == steps):
            value = _cached_feedback(model, feedback, cfg, device)
            control.window(value, 1, step, step, terminal=step == steps)
            train_mode(model, False)
        sync(device)
        monitoring_s += time.perf_counter() - t
        if progress is not None and (step % 100 == 0 or step == steps):
            progress(f'Cost diagnostic {condition}: {step}/{steps} steps')
    sync(device)
    elapsed = time.perf_counter() - begin
    state = control.state_dict() if control is not None else None
    return dict(elapsed_seconds=elapsed, train_compute_seconds=compute_s,
                monitoring_seconds=monitoring_s, transfer_seconds=transfer_s,
                other_seconds=elapsed - compute_s - monitoring_s - transfer_s,
                controller_windows=control.windows if control is not None else 0,
                feedback_evaluations=(1 + math.ceil(steps / cfg['control_interval']))
                    if control is not None else 0,
                controller_state_json_bytes=len(json.dumps(state, sort_keys=True).encode())
                    if state is not None else 0,
                controller_state=state,
                research_trace_rows=len(trace),
                research_trace_sha256=hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
                    if instrumented else None,
                research_batch_chain_sha256=batch_chain if instrumented else None)


def _tuning_costs(campaign_root):
    from common import read
    results = read(Path(campaign_root) / 'RESULTS.json')
    rows = [r for r in results['results'] if r['stage'] == 'tuning']
    groups = {}
    for row in rows:
        family = row['method_spec']['family']
        out = groups.setdefault(family, dict(method_runs=0, training_seconds=0.,
                                             acquisition_seconds=0., optimizer_steps=0))
        out['method_runs'] += 1
        out['training_seconds'] += row['training_seconds']
        out['acquisition_seconds'] += row['acquisition_seconds']
        out['optimizer_steps'] += row['training_steps']
    for group in groups.values():
        group['recorded_training_and_acquisition_hours'] = (
            group['training_seconds'] + group['acquisition_seconds']) / 3600
    return dict(source='Completed campaign RESULTS.json', families=groups,
                includes_shared_initialization=False,
                includes_all_evaluation_checkpoint_and_setup_time=False,
                lite_development_search_cost='Unknown; frozen settings are not evidence of zero prior development cost',
                engineering_effort='Not measured by runtime; includes implementation, state handling, testing and maintenance')


def _summary(rows):
    summaries = {}
    for condition in CONDITIONS:
        rr = [r for r in rows if r['condition'] == condition]
        summaries[condition] = {}
        for key in ('elapsed_seconds', 'train_compute_seconds', 'monitoring_seconds',
                    'transfer_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes',
                    'controller_state_json_bytes'):
            v = [float(r[key]) for r in rr]
            summaries[condition][key] = dict(mean=statistics.mean(v),
                sd=statistics.stdev(v) if len(v) > 1 else 0., median=statistics.median(v),
                min=min(v), max=max(v), repetitions=len(v))
    contrasts = []
    for implementation in ('streamlined', 'instrumented'):
        for target, reference in (('fixed', 'entropy'), ('lite', 'fixed'), ('lite', 'entropy')):
            name_a, name_b = f'{implementation}_{target}', f'{implementation}_{reference}'
            a = {r['repetition']: r['elapsed_seconds'] for r in rows if r['condition'] == name_a}
            b = {r['repetition']: r['elapsed_seconds'] for r in rows if r['condition'] == name_b}
            pct = [100 * (a[k] / b[k] - 1) for k in sorted(a)]
            diff = [a[k] - b[k] for k in sorted(a)]
            contrasts.append(dict(target=name_a, reference=name_b,
                paired_percent_overhead=pct, paired_seconds_difference=diff,
                mean_percent_overhead=statistics.mean(pct),
                sd_percent_overhead=statistics.stdev(pct),
                mean_seconds_difference=statistics.mean(diff),
                timing_repetitions_are_not_independent_training_seeds=True))
    for treatment in ('entropy', 'fixed', 'lite'):
        a = {r['repetition']: r['elapsed_seconds'] for r in rows
             if r['condition'] == f'instrumented_{treatment}'}
        b = {r['repetition']: r['elapsed_seconds'] for r in rows
             if r['condition'] == f'streamlined_{treatment}'}
        pct = [100 * (a[k] / b[k] - 1) for k in sorted(a)]
        contrasts.append(dict(target=f'instrumented_{treatment}', reference=f'streamlined_{treatment}',
            paired_percent_overhead=pct, mean_percent_overhead=statistics.mean(pct),
            sd_percent_overhead=statistics.stdev(pct),
            timing_repetitions_are_not_independent_training_seeds=True))
    return dict(conditions=summaries, paired_descriptive_contrasts=contrasts,
                hypothesis_tests='None: repeated timings on one checkpoint are descriptive')


def run(ctx):
    """Run 6 x 5 finite paired timings, then discard every modified parameter."""
    import numpy as np
    import torch
    from common import atomic_json, digest, read, sha
    from support import BalancedBatches, evaluation_batches, loader, optimizer_for, seed_all, sync
    p = ctx.protocol['cost']
    if (p['warmup_steps'], p['measured_steps'], p['repetitions'], p['cached_batches']) != (20, 200, 5, 8):
        raise ValueError('Cost protocol must remain the fixed 20/200/5/8 design')
    if ctx.cfg['control_interval'] != 100:
        raise ValueError('Cost design assumes the original 100-step feedback interval')
    out = Path(ctx.root) / 'cost'
    out.mkdir(parents=True, exist_ok=True)
    records = [r for r in ctx.records if r['stage'] == 'confirmation' and
               r['seed'] == 62001 and r['method'] == 'entropy']
    if len(records) != 1:
        raise ValueError('Need the predefined seed62001 entropy final checkpoint')
    record = records[0]
    campaign = Path(ctx.campaign_root)
    sd = campaign / 'confirmation' / 'seed62001'
    with np.load(sd / 'splits.npz', allow_pickle=False) as z:
        labeled = z['initial'].copy()
        holdout = z['holdout'].copy()
    original_files = [Path(record['path']), campaign/'SELECTION.json', campaign/'RESULTS.json',
                      sd/'splits.npz', sd/'targets.json']
    for phase in range(1, ctx.cfg['rounds'] + 1):
        path = sd / 'entropy' / 'acquisitions' / f'round{phase}' / 'selection.npz'
        original_files.append(path)
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['previous_labeled'], labeled) or not np.array_equal(z['control_holdout'], holdout):
                raise ValueError('Entropy acquisition input chain differs')
            labeled = np.sort(np.r_[labeled, z['selected']])
    if len(np.unique(labeled)) != len(labeled) or np.intersect1d(labeled, holdout).size:
        raise ValueError('Invalid final label/holdout partition')
    before = {str(path): sha(path) for path in original_files}
    if before[str(Path(record['path']))] != record['sha256']:
        raise ValueError('Benchmark checkpoint does not match inspected record')
    cfg = copy.deepcopy(ctx.cfg)
    cfg['eval_batch'] = p['feedback_batch']
    cfg['lambda_fixed'] = float(read(campaign / 'SELECTION.json')['fixed']['coefficient'])
    targets = read(sd / 'targets.json')['targets']
    cache_seed = 903170
    seed_all(cache_seed)
    start_cache = time.perf_counter()
    sampler = BalancedBatches(labeled, ctx.cat.labels, cache_seed, p['cached_batches'],
                              cfg['batch_classes'], cfg['samples_per_class'])
    cache_ids = sorted(set(int(idx) for batch in sampler for _, idx in batch) | set(map(int, holdout)))
    cache_images = {}
    for idx in cache_ids:
        image = ctx.cat.rows['train'][idx]
        path = ctx.cat.root / image['path']
        if sha(path) != image['sha256']:
            raise ValueError('Cost input training image bytes changed: ' + image['path'])
        cache_images[image['path']] = image['sha256']
    batches = list(loader(ctx.cat, 'train', sampler, cache_seed, p['workers'], True))
    feedback_batches = list(loader(ctx.cat, 'train', evaluation_batches(holdout,
        cfg['eval_batch']), cache_seed, p['workers'], False))
    feedback = (torch.cat([b[0] for b in feedback_batches]),
                torch.cat([b[1] for b in feedback_batches]))
    del feedback_batches
    cache_s = time.perf_counter() - start_cache
    if any(b[0].dtype != torch.uint8 for b in batches) or feedback[0].dtype != torch.uint8:
        raise ValueError('Image caches must stay bounded uint8 CPU tensors')
    for x, y, ids in batches:
        _, counts = torch.unique(y, return_counts=True)
        if len(y) != cfg['batch_classes'] * cfg['samples_per_class'] or not torch.all(counts == cfg['samples_per_class']):
            raise ValueError('Cached image batch violates the balanced sampler')
    cache_hash = _tensor_hash({f'batch_{i}_{k}': value for i, batch in enumerate(batches)
        for k, value in zip(('images', 'labels', 'ids'), batch)} |
        {'feedback_images': feedback[0], 'feedback_labels': feedback[1]})
    model = ctx.load_model(record)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    weights_hash = _tensor_hash(initial)
    windows = math.ceil(p['measured_steps'] / cfg['control_interval'])
    plan = dict(binding=ctx.binding, source_checkpoint_sha256=record['sha256'],
        source_model_tensor_sha256=weights_hash, targets_sha256=before[str(sd/'targets.json')],
        selection_sha256=before[str(campaign/'SELECTION.json')], fixed_coefficient=cfg['lambda_fixed'],
        warmup_steps=p['warmup_steps'], measured_steps=p['measured_steps'],
        repetitions=p['repetitions'], conditions=list(CONDITIONS),
        order_by_repetition=[condition_order(r) for r in range(p['repetitions'])],
        cached_training_batches=len(batches), cached_training_batch_size=len(batches[0][1]),
        cached_batch_ids=[b[2].tolist() for b in batches], training_labeled_count=len(labeled),
        feedback_image_count=len(holdout), feedback_ids=holdout.tolist(),
        cache_tensor_sha256=cache_hash, cache_construction_seconds=cache_s,
        cache_image_file_count=len(cache_images), cache_image_manifest_digest=digest(cache_images),
        cache_image_bytes_verified_before_decode=True,
        cpu_cache_bytes=sum(v.numel()*v.element_size() for b in batches for v in b) +
            sum(v.numel()*v.element_size() for v in feedback),
        cache_policy='Eight fixed augmented training minibatches are reused cyclically; feedback images use original deterministic center crops.',
        start_policy='Same final entropy weights, seed62001 anchors and fresh SGD state in every condition. Warmup is discarded; weights, optimizer, RNG and controller are reset before measurement.',
        precision='Original bf16 forward autocast, fp32 parameters and geometry; fp32 feedback inference; frozen BatchNorm running statistics.',
        optimizer_policy=f'Fresh original SGD parameter groups/momentum/Nesterov/decay; constant learning rate {cfg["lr_round"]}. No cosine schedule is extrapolated from this short timing diagnostic.',
        measured_boundary='Synchronized training, CPU-to-GPU transfer and controller monitoring including initial/100/200-step holdout CE; excludes cache creation, checkpoint loading/saving, acquisition and evaluation export.',
        lite_ce_policy='Faithful original Control path retained, including holdout CE even though Lite coefficient formula does not depend on CE.',
        streamlined_fixed='Exact geometry objective, without unused total/kappa statistics, record conversion, EMA or holdout evaluation.',
        research_instrumented='Full original geometry scalar monitoring, EMA, in-memory scalar trace and batch-chain hashing, plus holdout CE for every treatment; no checkpoint or CSV disk writes.',
        limitation='Short cached-batch timings on one backbone/checkpoint/hardware setup. They cannot be extrapolated into full-job speed, accuracy or tuning efficiency.')
    plan['plan_digest'] = digest({k:v for k,v in plan.items() if k != 'cache_construction_seconds'})
    atomic_json(out / 'PLAN.json', plan)
    rows = []
    for repetition in range(p['repetitions']):
        for position, condition in enumerate(condition_order(repetition)):
            row_path = out / f'repetition_{repetition:02d}_{condition}.json'
            if row_path.exists():
                saved = read(row_path)
                if saved['plan_digest'] != plan['plan_digest']:
                    raise ValueError('Existing cost repetition is bound to different inputs')
                rows.append(saved)
                continue
            ctx.progress(f'Cost diagnostic repetition {repetition+1}/{p["repetitions"]}: {condition}')
            model.load_state_dict(initial, strict=True)
            seed_all(903170 + repetition)
            opt = optimizer_for(model, cfg)
            ctrl = _make_control(condition, targets, cfg, ctx.device, windows)
            execute(model, opt, ctrl, condition, batches, feedback, targets, cfg,
                    ctx.device, p['warmup_steps'])
            # Warmup must not change the measured starting checkpoint or state.
            del opt, ctrl
            model.load_state_dict(initial, strict=True)
            model.zero_grad(set_to_none=True)
            seed_all(903170 + repetition)
            opt = optimizer_for(model, cfg)
            ctrl = _make_control(condition, targets, cfg, ctx.device, windows)
            if str(ctx.device).startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            result = execute(model, opt, ctrl, condition, batches, feedback, targets,
                             cfg, ctx.device, p['measured_steps'], ctx.progress)
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(ctx.device).startswith('cuda') else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if str(ctx.device).startswith('cuda') else 0,
                condition=condition, repetition=repetition, order_position=position,
                measured_steps=p['measured_steps'], measured_start_tensor_sha256=weights_hash,
                source_checkpoint_sha256=record['sha256'], plan_digest=plan['plan_digest'],
                final_transient_tensor_sha256=_tensor_hash(model.state_dict()))
            atomic_json(row_path, result)
            rows.append(result)
            del opt, ctrl
            model.zero_grad(set_to_none=True)
            gc.collect()
            if str(ctx.device).startswith('cuda'):
                torch.cuda.empty_cache()
    after = {str(path): sha(path) for path in original_files}
    if after != before:
        raise ValueError('Read-only benchmark source changed during execution')
    report = dict(binding=ctx.binding, status='COMPLETE', plan=plan,
        original_input_hashes_before=before, original_input_hashes_after=after,
        original_inputs_unchanged=True, rows=rows, summary=_summary(rows),
        tuning_cost=_tuning_costs(campaign),
        modified_models_and_optimizer_states_saved=False,
        outcomes_do_not_establish_accuracy_or_controller_superiority=True)
    atomic_json(out / 'COST_RESULTS.json', report)
    del model, initial, batches, feedback
    gc.collect()
    if str(ctx.device).startswith('cuda'):
        torch.cuda.empty_cache()
    return report
