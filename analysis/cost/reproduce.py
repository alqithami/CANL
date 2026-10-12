#!/usr/bin/env python3
"""Reconstruct completed CÆNL timing summaries on CPU (Python standard library)."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def summarize(values):
    return {'mean': statistics.mean(values), 'sd': statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, default=Path(__file__).resolve().parents[2] / 'results/2026-10-12-cost')
    parser.add_argument('--output', type=Path, required=True, help='A new output directory; existing paths are never overwritten.')
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a new output directory.')
    records = args.records.resolve()
    manifest = json.loads((records / 'SHA256SUMS.json').read_text())
    for name, expected in manifest.items():
        require(Path(name).name == name, 'Unsafe input manifest path.')
        actual = hashlib.sha256((records / name).read_bytes()).hexdigest()
        require(actual == expected, 'Input digest mismatch: ' + name)
    errors = {}

    def near(label, actual, expected, tolerance=1e-9):
        error = abs(actual - expected)
        require(error <= tolerance, f'{label}: reconstructed {actual}, expected {expected}.')
        errors[label] = error

    runtime = read_csv(records / 'feedback_runtime.csv')
    require(len(runtime) == 104, 'Expected 104 completed feedback method records.')
    keys = [(r['stage'], int(r['seed']), r['method']) for r in runtime]
    require(len(set(keys)) == len(keys), 'Duplicate feedback timing record.')
    groups = defaultdict(list)
    for row in runtime:
        for field in ['phase_duration_seconds', 'train_compute_seconds', 'acquisition_seconds']:
            row[field] = float(row[field])
            require(math.isfinite(row[field]) and row[field] > 0, 'Invalid timing scalar.')
        row['phase_plus_acquisition_seconds'] = row['phase_duration_seconds'] + row['acquisition_seconds']
        groups[row['stage'], row['backbone'], row['method']].append(row)
    expected_counts = {'tuning': 24, 'confirmation': 40, 'transfer': 40}
    require({stage: sum(r['stage'] == stage for r in runtime) for stage in expected_counts} == expected_counts, 'Unexpected stage counts.')
    expected_methods = {
        'tuning': {'fixed_0p003', 'fixed_0p010', 'fixed_0p030', 'fixed_0p100', 'linear_down_0p03', 'linear_down_0p10', 'linear_up_0p03', 'linear_up_0p10'},
        'confirmation': {'entropy', 'lite', 'tuned_fixed', 'tuned_schedule'},
        'transfer': {'entropy', 'lite', 'tuned_fixed', 'tuned_schedule'},
    }
    expected_seeds = {'tuning': set(range(41001, 41004)), 'confirmation': set(range(42001, 42011)), 'transfer': set(range(43001, 43011))}
    expected_backbones = {'tuning': 'resnet50', 'confirmation': 'resnet50', 'transfer': 'resnet18'}
    feedback_groups = []
    for stage in expected_counts:
        require({r['method'] for r in runtime if r['stage'] == stage} == expected_methods[stage], 'Unexpected feedback methods.')
    for (stage, backbone, method), rows in sorted(groups.items()):
        require(backbone == expected_backbones[stage], 'Unexpected feedback backbone.')
        require({int(r['seed']) for r in rows} == expected_seeds[stage], 'Missing or unexpected feedback seed.')
        result = {'stage': stage, 'backbone': backbone, 'method': method, 'n_seeds': len(rows)}
        for field in ['phase_duration_seconds', 'train_compute_seconds', 'acquisition_seconds', 'phase_plus_acquisition_seconds']:
            stats = summarize([r[field] for r in rows])
            result[field + '_mean'] = stats['mean']
            result[field + '_sd'] = stats['sd']
        result['phase_plus_acquisition_hours_mean'] = result['phase_plus_acquisition_seconds_mean'] / 3600
        feedback_groups.append(result)
    reference = json.loads((records / 'feedback_runtime_reference.json').read_text())
    for row in reference['mean_train_plus_acquisition_hours']:
        reconstructed = next(r for r in feedback_groups if (r['stage'], r['method']) == (row['stage'], row['method']))
        near('feedback/' + row['stage'] + '/' + row['method'], reconstructed['phase_plus_acquisition_hours_mean'], row['train_plus_acquisition_hours'])
    initial = json.loads((records / 'feedback_initializations.json').read_text())
    require(len(initial) == 23, 'Expected 23 shared initializations.')
    require(len({(r['stage'], r['seed']) for r in initial}) == 23, 'Duplicate initialization record.')
    allocation = []
    for stage in expected_counts:
        selected = [r for r in initial if r['stage'] == stage]
        require({r['seed'] for r in selected} == expected_seeds[stage], 'Unexpected initialization seeds.')
        initial_seconds = sum(r['phase_training_elapsed_seconds'] for r in selected)
        near('initialization/' + stage, initial_seconds / 3600, reference['initialization_training_hours_by_stage'][stage])
        method_seconds = sum(r['phase_plus_acquisition_seconds'] for r in runtime if r['stage'] == stage)
        allocation.append({
            'stage': stage, 'method_runs': expected_counts[stage], 'shared_initializations': len(selected),
            'postinitial_phase_plus_acquisition_hours': method_seconds / 3600,
            'shared_initialization_phase_hours': initial_seconds / 3600,
            'recorded_total_hours': (method_seconds + initial_seconds) / 3600,
        })
    near('tuning/phase_plus_acquisition_hours', allocation[0]['postinitial_phase_plus_acquisition_hours'], reference['tuning_total_train_plus_acquisition_hours'])

    micro = read_csv(records / 'c4_microbenchmark_raw.csv')
    require(len(micro) == 10, 'Expected five paired microbenchmark repeats.')
    cfg = json.loads((records / 'c4_microbenchmark_config.json').read_text())
    require((cfg['repeats'], cfg['warmup_steps'], cfg['measure_steps'], cfg['effective_batch_tokens']) == (5, 25, 250, 8192), 'Unexpected microbenchmark design.')
    grouped = defaultdict(list)
    paired = {}
    for row in micro:
        key = (int(row['repeat']), row['mode'])
        require(key not in paired, 'Duplicate microbenchmark observation.')
        paired[key] = row
        require(int(row['measured_optimizer_steps']) == 250 and int(row['tokens']) == 2048000, 'Unexpected microbenchmark workload.')
        require(int(row['policy_updates']) == 0, 'Microbenchmark is monitoring only, not a controller-policy benchmark.')
        for field in ['elapsed_seconds', 'tokens_per_second', 'milliseconds_per_step', 'peak_allocated_gib', 'peak_reserved_gib']:
            row[field] = float(row[field])
            require(math.isfinite(row[field]) and row[field] > 0, 'Invalid microbenchmark scalar.')
        near('micro/tokens_per_second/' + str(key), row['tokens_per_second'], int(row['tokens']) / row['elapsed_seconds'], 1e-7)
        grouped[row['mode']].append(row)
    require(set(paired) == {(i, mode) for i in range(1, 6) for mode in ['plain', 'monitor_only']}, 'Incomplete paired microbenchmark.')
    paired_rows = []
    for repeat in range(1, 6):
        plain = paired[repeat, 'plain']
        monitor = paired[repeat, 'monitor_only']
        require(int(plain['monitor_events']) == 0 and int(monitor['monitor_events']) == 10, 'Unexpected monitoring cadence.')
        paired_rows.append({
            'repeat': repeat, 'plain_seconds': plain['elapsed_seconds'],
            'monitor_seconds': monitor['elapsed_seconds'],
            'paired_time_increase_pct': 100 * (monitor['elapsed_seconds'] / plain['elapsed_seconds'] - 1),
            'peak_allocated_increase_gib': monitor['peak_allocated_gib'] - plain['peak_allocated_gib'],
        })
    micro_groups = []
    micro_ref = json.loads((records / 'c4_microbenchmark_reference.json').read_text())
    for mode in ['plain', 'monitor_only']:
        rows = grouped[mode]
        result = {'mode': mode, 'n_repeats': len(rows)}
        for field in ['elapsed_seconds', 'tokens_per_second', 'peak_allocated_gib', 'peak_reserved_gib']:
            stats = summarize([r[field] for r in rows])
            result[field + '_mean'] = stats['mean']
            result[field + '_sd'] = stats['sd']
        values = [r['paired_time_increase_pct'] for r in paired_rows] if mode == 'monitor_only' else [0.] * 5
        result['paired_time_increase_pct_mean'] = statistics.mean(values)
        result['paired_time_increase_pct_sd'] = statistics.stdev(values)
        micro_groups.append(result)
        old = next(r for r in micro_ref['grouped'] if r['mode'] == mode)
        aliases = {
            'elapsed_seconds_mean': 'elapsed_mean_s', 'elapsed_seconds_sd': 'elapsed_std_s',
            'tokens_per_second_mean': 'tokens_per_second_mean', 'tokens_per_second_sd': 'tokens_per_second_std',
            'peak_allocated_gib_mean': 'peak_allocated_gib_mean', 'peak_allocated_gib_sd': 'peak_allocated_gib_std',
            'paired_time_increase_pct_mean': 'overhead_vs_plain_mean_pct', 'paired_time_increase_pct_sd': 'overhead_vs_plain_std_pct',
        }
        for field, old_field in aliases.items():
            near('micro/' + mode + '/' + field, result[field], old[old_field], 1e-7)

    production = read_csv(records / 'c4_production_jobs.csv')
    require(len(production) == 24, 'Expected 24 C4 production timing records.')
    require(len({(r['method'], int(float(r['seed']))) for r in production}) == 24, 'Duplicate C4 production record.')
    require({r['method'] for r in production} == {'baseline', 'fixed_dcr', 'macc_lite', 'full_macc'}, 'Unexpected C4 production methods.')
    loop_reference = json.loads((records / 'c4_training_loop.json').read_text())
    loop = loop_reference['rows']
    require(len(loop) == 24, 'Expected 24 successful C4 training-loop summaries.')
    loop_by_key = {(r['method'], r['seed']): r for r in loop}
    require(len(loop_by_key) == 24, 'Duplicate C4 training-loop summary.')
    require(set(loop_by_key) == {(r['method'], int(float(r['seed']))) for r in production}, 'Training-loop and task-job seed sets differ.')
    for row in production:
        key = (row['method'], int(float(row['seed'])))
        summary = loop_by_key[key]
        require(summary['steps'] == int(float(row['steps'])) and summary['tokens_trained'] == int(float(row['tokens_trained'])), 'Training-loop and task-job workloads differ.')
        digest = summary['source_summary_sha256']
        require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Invalid source-summary digest.')
        require(math.isfinite(summary['train_time_s']) and 0 < summary['train_time_s'] <= summary['wall_time_s'], 'Invalid training-loop timing boundary.')
        near('production/source_jobwall/' + row['job_id'], summary['wall_time_s'], float(row['wall_time_s']))
    production_groups = []
    for method in ['baseline', 'fixed_dcr', 'macc_lite', 'full_macc']:
        rows = [r for r in production if r['method'] == method]
        require({int(float(r['seed'])) for r in rows} == set(range(301, 307)), 'Unexpected production seeds.')
        for row in rows:
            require(int(float(row['steps'])) == 12207 and int(float(row['tokens_trained'])) == 99999744, 'Unequal C4 production workload.')
            require(int(row['attempt']) == 1 and int(row['attempt_count']) == 1, 'Unexpected retried C4 production job.')
            require(row['task'] == 'language_c4' and row['model'] == 'gpt2', 'Unexpected C4 production task/model.')
            near('production/attempt_wall/' + row['job_id'], float(row['attempt_wall_time_s']), float(row['wall_time_s']))
        stats = summarize([float(r['wall_time_s']) for r in rows])
        loop_stats = summarize([r['train_time_s'] for r in loop if r['method'] == method])
        production_groups.append({'method': method, 'n_seeds': len(rows), 'successful_training_loop_seconds_mean': loop_stats['mean'], 'successful_training_loop_seconds_sd': loop_stats['sd'], 'task_job_wall_seconds_mean': stats['mean'], 'task_job_wall_seconds_sd': stats['sd']})
    baseline = production_groups[0]['task_job_wall_seconds_mean']
    loop_baseline = production_groups[0]['successful_training_loop_seconds_mean']
    published_rounding = {'fixed_dcr': 7.88, 'macc_lite': 7.68, 'full_macc': 16.62}
    for row in production_groups:
        row['ratio_of_mean_wall_times_increase_pct'] = 100 * (row['task_job_wall_seconds_mean'] / baseline - 1)
        row['ratio_of_mean_training_loop_times_increase_pct'] = 100 * (row['successful_training_loop_seconds_mean'] / loop_baseline - 1)
        method = row['method']
        near('production/training_loop_mean/' + method, row['successful_training_loop_seconds_mean'], loop_reference['train_time_means_s'][method])
        near('production/training_loop_ratio/' + method, row['ratio_of_mean_training_loop_times_increase_pct'], loop_reference['train_time_overhead_pct'][method])
        if method in published_rounding:
            near('production/published_training_loop_rounding/' + method, round(row['ratio_of_mean_training_loop_times_increase_pct'], 2), published_rounding[method])

    output = args.output
    output.mkdir(parents=True)
    for filename, rows in [
        ('feedback_group_costs.csv', feedback_groups), ('feedback_stage_allocation.csv', allocation),
        ('c4_microbenchmark_pairs.csv', paired_rows), ('c4_microbenchmark_groups.csv', micro_groups),
        ('c4_production_costs.csv', production_groups),
    ]:
        write_csv(output / filename, rows)
    write_json(output / 'cost_summary.json', {
        'feedback_groups': feedback_groups, 'feedback_stage_allocation': allocation,
        'c4_microbenchmark_pairs': paired_rows, 'c4_microbenchmark_groups': micro_groups,
        'c4_production_costs': production_groups,
        'measurement_boundaries': {
            'feedback_phase_plus_acquisition': 'Sum of five recorded post-initialization phase-duration timers plus acquisition timers. Phase duration already includes training and within-phase operations; synchronized train_compute is an overlapping component and must not be added again. Shared initialization is recorded once per seed when forming stage totals.',
            'feedback_research_instrumentation': 'All four confirmation/transfer methods compute matched-window holdout feedback, including fixed/entropy methods. Lite retains its actual held-out cross-entropy observation even though that metric does not enter its coefficient action. These timings do not compare with lean fixed or lean unregularized deployment.',
            'c4_microbenchmark': 'Five randomized-order plain/monitor-only pairs with the same fixed token tensors, 25 untimed warmup steps and 250 measured optimizer steps. Initialization and model/data loading excluded. No learned policy update is measured.',
            'c4_production': 'Two distinct recorded boundaries for six matched-workload seeds per method: compute.train_time_s is successful training-loop time, while wall_time_s is broader successful task-job wall time. Percentages are ratios of group means for the respective boundary against the monitored baseline, not means of paired seed percentages. The broader job timer includes the task driver. Neither can be combined with the separate compute microbenchmark.',
            'not_measured': 'These records do not establish total deployment overhead, isolated controller memory, earlier method-development costs, economic savings, or a universal sub-10-percent bound.',
        },
    })
    write_json(output / 'VERIFICATION.json', {
        'status': 'PASS', 'input_sha256_verified': manifest,
        'counts': {'feedback_methods': 104, 'feedback_initializations': 23, 'c4_microbenchmark_observations': 10, 'c4_production_jobs': 24, 'c4_training_loop_summaries': 24},
        'reference_comparisons': len(errors), 'maximum_absolute_reference_error': max(errors.values()),
        'reference_error_units': 'Mixed units; each comparison uses its explicit tolerance. Throughput and microbenchmark checks allow 1e-7; feedback-hour and production-timer/reference checks allow 1e-9.',
        'reference_errors': errors,
        'scope': 'Input digests, workload/count/seed consistency and arithmetic against archived summaries. No GPU timing, training, inference or missing historical raw timing records were recreated.',
    })
    print(json.dumps({'status': 'PASS', 'reference_comparisons': len(errors), 'tuning_recorded_hours': allocation[0]['recorded_total_hours'], 'monitor_only_paired_increase_pct': micro_groups[1]['paired_time_increase_pct_mean']}))


if __name__ == '__main__':
    main()
