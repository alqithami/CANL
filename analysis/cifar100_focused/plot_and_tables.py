#!/usr/bin/env python3
"""Export the complete frozen CIFAR100 test grid, tables, and column-size plot.

This program reads released records only. It neither trains nor selects models.
Usage: python3 plot_and_tables.py --input RESULTS_INPUT.json \
  --analysis ANALYSIS.json --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

METHODS = ('ce', 'ratio_dcr', 'full_dcr', 'plain_fisher')
SEEDS = tuple(range(92001, 92011))
COUNTS = (4500, 9000, 18000, 27000)
WEIGHTS = np.array((.1, .3, .4, .2))
LABELS = {'ce': 'CE', 'ratio_dcr': 'Ratio DCR', 'full_dcr': 'Full DCR',
          'plain_fisher': 'Fisher'}
TEX_LABELS = {'ce': 'CE', 'ratio_dcr': 'Ratio DCR', 'full_dcr': 'Full DCR',
              'plain_fisher': 'Fisher'}
SHORT = {'ce': 'C', 'ratio_dcr': 'R', 'full_dcr': 'D', 'plain_fisher': 'F'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ptex(value):
    value = float(value)
    if value == 0:
        return '0'
    if value < 1e-4:
        mantissa, exponent = f'{value:.2e}'.split('e')
        return mantissa.rstrip('0').rstrip('.') + r'\times10^{' + str(int(exponent)) + '}'
    return f'{value:.5f}'.rstrip('0').rstrip('.')


def pm(mean, sd):
    return f'${mean:.3f}\\pm{sd:.3f}$'


def table(caption, label, columns, header, rows, footnote=None):
    result = ['\\begin{table}[tbp]', '\\centering\\footnotesize',
              '\\setlength{\\tabcolsep}{2pt}', '\\caption{' + caption + '}',
              '\\label{' + label + '}',
              '\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}' + columns + '@{}}',
              '\\toprule', header + r' \\', '\\midrule']
    result.extend(row + r' \\' for row in rows)
    result.extend(['\\bottomrule', '\\end{tabular*}'])
    if footnote:
        result.extend(['\\par\\smallskip', '\\begin{minipage}{\\linewidth}\\scriptsize',
                       footnote, '\\end{minipage}'])
    result.extend(['\\end{table}', ''])
    return '\n'.join(result)


def load_grid(input_path, analysis_path):
    data = json.loads(Path(input_path).read_text())
    analysis = json.loads(Path(analysis_path).read_text())
    if data.get('status') != 'COMPLETED_VERIFIED' or data.get('evaluation_partition') != 'official_test' or data.get('test_release_verified') is not True:
        raise ValueError('Only completed explicitly released official-test records may be exported')
    if analysis.get('status') != 'CONFIRMATION_ANALYSIS_COMPLETE' or analysis.get('seeds') != list(SEEDS):
        raise ValueError('Frozen ten-seed analysis required')
    if data.get('protocol_sha256') != analysis.get('protocol_sha256') or data.get('source_binding_sha256') != analysis.get('source_binding_sha256'):
        raise ValueError('Input and analysis binding mismatch')
    records = data['records']
    if len(records) != 160:
        raise ValueError('Exactly160 records required')
    lookup = {}
    for row in records:
        key = (row['method'], row['seed'], row['label_count'])
        if key in lookup:
            raise ValueError('Duplicate test-grid record')
        if row.get('evaluation_partition') != 'official_test' or row.get('status') != 'COMPLETED_VERIFIED':
            raise ValueError('Unverified or non-test record')
        value = row['accuracy_percent']
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('Invalid accuracy')
        lookup[key] = row
    if set(lookup) != set(itertools.product(METHODS, SEEDS, COUNTS)):
        raise ValueError('Missing or changed seed/method/budget grid')
    curves = {m: np.array([[lookup[m, s, n]['accuracy_percent'] for n in COUNTS]
                           for s in SEEDS], dtype=float) for m in METHODS}
    aulc = {m: curves[m] @ WEIGHTS for m in METHODS}
    for method in METHODS:
        saved = analysis['methods'][method]
        if not np.allclose(aulc[method], saved['seed_AULC'], rtol=0, atol=1e-10):
            raise ValueError('Saved seed AULCs do not match complete grid')
        if not np.isclose(aulc[method].mean(), saved['mean_AULC'], rtol=0, atol=1e-10):
            raise ValueError('Saved AULC mean mismatch')
        if not np.isclose(aulc[method].std(ddof=1), saved['seed_SD_AULC'], rtol=0, atol=1e-10):
            raise ValueError('Saved AULC SD mismatch')
        if not np.allclose(curves[method].mean(axis=0), saved['mean_accuracy_percent_by_budget'], rtol=0, atol=1e-10):
            raise ValueError('Saved mean curve mismatch')
    if [(x['target'], x['control']) for x in analysis['planned_contrasts']] != [('ratio_dcr', m) for m in METHODS if m != 'ratio_dcr']:
        raise ValueError('Three frozen primary contrasts required')
    return data, analysis, lookup, curves, aulc


def export(args):
    data, analysis, lookup, curves, aulc = load_grid(args.input, args.analysis)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    ordered = [lookup[m, s, n] for m in METHODS for s in SEEDS for n in COUNTS]
    write_csv(output / 'all160_official_test_accuracies.csv', ordered)
    summaries = []
    for m in METHODS:
        for i, count in enumerate(COUNTS):
            summaries.append({'method': m, 'label_count': count,
                              'additional_development_labels': 5000, 'n_seeds': 10,
                              'test_accuracy_mean_percent': float(curves[m][:, i].mean()),
                              'test_accuracy_sample_sd_pp': float(curves[m][:, i].std(ddof=1)),
                              'normalized_AULC_mean_pp': float(aulc[m].mean()),
                              'normalized_AULC_sample_sd_pp': float(aulc[m].std(ddof=1))})
    write_csv(output / 'method_budget_summary.csv', summaries)
    seed_rows = [{'seed': s, **{m: float(aulc[m][i]) for m in METHODS}}
                 for i, s in enumerate(SEEDS)]
    write_csv(output / 'ten_seed_AULC.csv', seed_rows)
    nonmonotonic = []
    for method in METHODS:
        differences = curves[method][:, -1] - curves[method][:, -2]
        nonmonotonic.append({
            'method': method,
            'mean_27k_minus_18k_pp': float(differences.mean()),
            'negative_seed_changes': int(np.count_nonzero(differences < 0)),
            'seed_changes': {str(s): float(differences[i]) for i, s in enumerate(SEEDS)},
            '27k_seed_values': {str(s): float(curves[method][i, -1]) for i, s in enumerate(SEEDS)},
            '27k_min': float(curves[method][:, -1].min()),
            '27k_max': float(curves[method][:, -1].max()),
        })
    (output / 'NONMONOTONIC_BUDGET_CHECK.json').write_text(json.dumps(nonmonotonic, indent=2) + '\n')
    contrasts = []
    for row in analysis['planned_contrasts']:
        sensitivity = row['paired_t_sensitivity']
        contrasts.append({'target': row['target'], 'control': row['control'],
                          'mean_AULC_difference_pp': sensitivity['mean_difference_pp'],
                          'ci95_unadjusted_lower_pp': sensitivity['ci95_unadjusted_pp'][0],
                          'ci95_unadjusted_upper_pp': sensitivity['ci95_unadjusted_pp'][1],
                          'exact_sign_flip_p_raw': row['exact_sign_flip_p_raw'],
                          'exact_sign_flip_p_Holm': row['exact_sign_flip_p_Holm'],
                          'paired_t_p_raw': sensitivity['p_raw'],
                          'paired_t_p_Holm_separate_family': sensitivity['p_Holm_separate_sensitivity_family']})
    write_csv(output / 'all_three_planned_contrasts.csv', contrasts)
    if data.get('costs', {}).get('records'):
        costs = data['costs']['records']
        write_csv(output / 'all40_method_seed_job_costs.csv', costs)
        cost_summary = []
        for method in METHODS:
            selected = [r for r in costs if r['method'] == method]
            if len(selected) != 10:
                raise ValueError('Complete40 method-seed costs required')
            for field in ('training_seconds', 'end_to_end_seconds'):
                values = np.array([r[field] for r in selected], dtype=float)
                cost_summary.append({'method': method, 'recorded_field': field,
                                     'reported_boundary': 'training time' if field == 'training_seconds' else 'summed job time',
                                     'mean_seconds_over_four_budgets': float(values.mean()),
                                     'sample_sd_seconds': float(values.std(ddof=1)),
                                     'scope': data['costs']['scope']})
        write_csv(output / 'descriptive_cost_summary.csv', cost_summary)

    rows = [f"{TEX_LABELS[m]} & {pm(aulc[m].mean(), aulc[m].std(ddof=1))} & {pm(curves[m][:, -1].mean(), curves[m][:, -1].std(ddof=1))}" for m in METHODS]
    (output / 'table_cifar100_summary.tex').write_text(table(
        'Separate CIFAR-100 official-test confirmation. Mean $\\pm$ sample SD over ten paired seeds. AULC is normalized over 4,500--27,000 acquired training labels; 5,000 labeled development images are additional.',
        'tab:cifar100_summary', 'lrr', 'Recipe & AULC & 27,000-label top-1 (\\%)', rows))
    primary_rows = []
    sensitivity_rows = []
    for row in contrasts:
        contrast = f"$R-{SHORT[row['control']]}$"
        ci = f"$[{row['ci95_unadjusted_lower_pp']:+.3f},{row['ci95_unadjusted_upper_pp']:+.3f}]$"
        primary_rows.append(f"{contrast} & ${row['mean_AULC_difference_pp']:+.3f}$ & {ci} & ${ptex(row['exact_sign_flip_p_raw'])}$ & ${ptex(row['exact_sign_flip_p_Holm'])}$")
        sensitivity_rows.append(f"{contrast} & ${ptex(row['paired_t_p_raw'])}$ & ${ptex(row['paired_t_p_Holm_separate_family'])}$")
    (output / 'table_cifar100_primary.tex').write_text(table(
        'All three prespecified CIFAR-100 AULC contrasts. Differences and individual, unadjusted paired-$t$ 95\\% intervals are in percentage points. Primary exact sign-flip $p$-values receive Holm correction across these three contrasts. $R$: ratio DCR; $C$: CE; $D$: full DCR; $F$: Fisher.',
        'tab:cifar100_primary', 'lrrrr', 'Contrast & $\\Delta$ & 95\\% CI & Exact $p$ & Holm $p$', primary_rows))
    (output / 'table_cifar100_sensitivity.tex').write_text(table(
        'CIFAR-100 paired-$t$ sensitivity analysis. Holm correction is separate from the primary exact-test family. Abbreviations follow Table~\\ref{tab:cifar100_primary}.',
        'tab:cifar100_sensitivity', 'lrr', 'Contrast & Raw $p$ & Holm $p$', sensitivity_rows))
    seed_tex = [str(row['seed']) + ' & ' + ' & '.join(f"${row[m]:.3f}$" for m in METHODS) for row in seed_rows]
    (output / 'table_cifar100_seeds.tex').write_text(table(
        'CIFAR-100 normalized AULC by independent confirmation seed. Every entry combines all four training-label budgets. All values use official-test accuracy percentages.',
        'tab:cifar100_seeds', 'lrrrr', 'Seed & CE & Ratio DCR & Full DCR & Fisher', seed_tex))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    styles = {'ce': dict(color='black', marker='o', linestyle='-'),
              'ratio_dcr': dict(color='#176B73', marker='s', linestyle='-'),
              'full_dcr': dict(color='#666666', marker='^', linestyle='--'),
              'plain_fisher': dict(color='black', marker='D', linestyle=':')}
    for method in METHODS:
        ax.errorbar(np.array(COUNTS) / 1000, curves[method].mean(axis=0),
                    yerr=curves[method].std(axis=0, ddof=1), label=LABELS[method],
                    linewidth=1.1, markersize=3.4, markeredgewidth=.65,
                    capsize=2, elinewidth=.6, **styles[method])
    ax.set_xticks(np.array(COUNTS) / 1000, ['4.5', '9', '18', '27'])
    ax.set_xlabel('Acquired training labels (thousands)', labelpad=3)
    ax.set_ylabel('Official-test top-1 accuracy (%)', labelpad=3)
    ax.grid(axis='y', linewidth=.45, color='#dddddd', zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=.6)
    ax.margins(x=.035, y=.07)
    ax.legend(loc='lower left', bbox_to_anchor=(-.02, 1.01), ncol=2,
              frameon=False, borderaxespad=0, columnspacing=1.2, handlelength=1.7,
              labelspacing=.35, handletextpad=.45)
    fig.subplots_adjust(left=.16, right=.99, bottom=.18, top=.80)
    fig.savefig(output / 'figure_cifar100_confirmation.pdf')
    fig.savefig(output / 'figure_cifar100_confirmation.png', dpi=200)
    plt.close(fig)
    manifest = {'input_sha256': sha(args.input), 'analysis_sha256': sha(args.analysis),
                'source_sha256': sha(__file__), 'n_records': 160,
                'n_paired_seeds': 10, 'error_bars': 'sample standard deviation across paired seeds, not confidence intervals',
                'figure_inches': [3.35, 2.65],
                'files': {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'COMPLETE_GRID_EXPORT_VERIFIED', 'output': str(output), 'records': 160, 'files': len(manifest['files'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    export(parser.parse_args())
