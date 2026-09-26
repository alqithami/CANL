"""Fixed, entity-bound styling for every figure (colour-blind-validated categorical palette).

Colours follow the *method*, never its rank in a particular plot; line styles give a
secondary encoding so identity never rests on colour alone.  Human-readable names are
used consistently across tables and figures.
"""
from __future__ import annotations

PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "gray": "#7a7975",
    "black": "#1b1b1b",
}

METHOD_STYLE: dict[str, dict] = {
    "random": {"color": PALETTE["gray"], "ls": "-", "label": "Random"},
    "entropy": {"color": PALETTE["orange"], "ls": "-", "label": "Entropy"},
    "margin": {"color": PALETTE["yellow"], "ls": "-", "label": "Margin"},
    "least_confidence": {"color": PALETTE["yellow"], "ls": ":", "label": "Least confidence"},
    "power_margin": {"color": PALETTE["yellow"], "ls": "--", "label": "Power-Margin (SBA)"},
    "coreset": {"color": PALETTE["magenta"], "ls": "-", "label": "CoreSet"},
    "badge": {"color": PALETTE["aqua"], "ls": "-", "label": "BADGE"},
    "bald_mcd": {"color": PALETTE["green"], "ls": "-", "label": "BALD (MC feat. dropout)"},
    "noise_stability": {"color": PALETTE["red"], "ls": "-", "label": "Noise Stability"},
    "caenl_acq_only": {"color": PALETTE["blue"], "ls": "--", "label": "CAENL acquisition only (no DCR)"},
    "dcr_fixed_random": {"color": PALETTE["gray"], "ls": "--", "label": "Random + fixed DCR"},
    "dcr_lite_random": {"color": PALETTE["gray"], "ls": ":", "label": "Random + MACC-Lite DCR"},
    "caenl_fixed_dcr": {"color": PALETTE["blue"], "ls": ":", "label": "CAENL + fixed DCR"},
    "caenl_macc_lite": {"color": PALETTE["blue"], "ls": "-", "label": "CAENL (MACC-Lite)"},
    "caenl_full_macc": {"color": PALETTE["violet"], "ls": "-", "label": "CAENL (Full MACC)"},
    "entropy_macc_lite": {"color": PALETTE["orange"], "ls": "--", "label": "Entropy + MACC-Lite DCR"},
    "at_random": {"color": PALETTE["red"], "ls": "-.", "label": "PGD-AT + random (reference)"},
    "at_caenl_macc_lite": {"color": PALETTE["blue"], "ls": "-.", "label": "PGD-AT + CAENL (reference)"},
    "supervised_full": {"color": PALETTE["black"], "ls": "-.", "label": "Full supervision (100%)"},
    "baseline": {"color": PALETTE["gray"], "ls": "-", "label": "Baseline (no DCR)"},
    "fixed_dcr": {"color": PALETTE["orange"], "ls": "-", "label": "Fixed DCR"},
    "macc_lite": {"color": PALETTE["blue"], "ls": "-", "label": "MACC-Lite"},
    "full_macc": {"color": PALETTE["violet"], "ls": "-", "label": "Full MACC"},
    "fixed_r16_4_l0005": {"color": PALETTE["orange"], "ls": "-", "label": "Fixed DCR (r=16/4, $\\lambda$=0.005)"},
    "fixed_r16_4_l001": {"color": PALETTE["yellow"], "ls": "-", "label": "Fixed DCR (r=16/4, $\\lambda$=0.01)"},
    "fixed_r32_8_l0005": {"color": PALETTE["magenta"], "ls": "-", "label": "Fixed DCR (r=32/8, $\\lambda$=0.005)"},
    "macc_lite_r16_4": {"color": PALETTE["blue"], "ls": "-", "label": "MACC-Lite (r=16/4)"},
    "full_macc_r16_4": {"color": PALETTE["violet"], "ls": "-", "label": "Full MACC (r=16/4)"},
    "fixed_dcr_selected": {"color": PALETTE["orange"], "ls": "-", "label": "Fixed DCR (selected)"},
    "macc_lite_selected": {"color": PALETTE["blue"], "ls": "-", "label": "MACC-Lite (selected)"},
    "full_macc_selected": {"color": PALETTE["violet"], "ls": "-", "label": "Full MACC (selected)"},
}

LAYER_STYLE = {"layer1": PALETTE["yellow"], "layer2": PALETTE["aqua"], "layer3": PALETTE["orange"], "layer4": PALETTE["blue"]}

# Order used in tables (baselines first, CAENL family, references)
METHOD_ORDER = [
    "random", "entropy", "margin", "least_confidence", "power_margin", "coreset", "badge", "bald_mcd", "noise_stability",
    "caenl_acq_only", "dcr_fixed_random", "dcr_lite_random", "caenl_fixed_dcr", "caenl_macc_lite", "caenl_full_macc", "entropy_macc_lite",
    "caenl_fixed_dcr_l0.01", "caenl_fixed_dcr_l0.1", "caenl_fixed_dcr_l0.3", "caenl_fixed_dcr_l1", "caenl_fixed_dcr_l3", "caenl_fixed_dcr_l10",
    "at_random", "at_caenl_macc_lite", "supervised_full",
    "baseline", "fixed_dcr", "macc_lite", "full_macc",
    "fixed_r16_4_l0005", "fixed_r16_4_l001", "fixed_r32_8_l0005", "macc_lite_r16_4", "full_macc_r16_4",
    "fixed_dcr_selected", "macc_lite_selected", "full_macc_selected",
]

DATASET_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tiny_imagenet": "Tiny-ImageNet", "imagenet100": "ImageNet-100", "imagenet1k": "ImageNet-1K", "synthetic": "Synthetic"}
METRIC_LABEL = {
    "test_acc": "Clean acc. (%)",
    "aa_robust_acc": "AutoAttack robust acc. (%)",
    "apgd_ce_subset_acc": "APGD-CE robust acc. (%)",
    "pgd_subset_acc": "PGD-20 robust acc. (%)",
    "test_ece": "ECE",
    "test_loss": "Test NLL",
    "corruption_mean_acc": "Corruption acc. (%)",
    "aulc_test_acc": "AULC clean acc. (%)",
    "aulc_apgd_acc": "AULC APGD-CE acc. (%)",
    "validation_nll": "Val. NLL",
    "perplexity": "Perplexity",
    "token_ece": "Token ECE",
    "fid": "FID",
    "kid": "KID x 1e3",
    "inception_score": "IS",
    "cider_d": "CIDEr-D",
    "bleu4": "BLEU-4",
    "rouge_l": "ROUGE-L",
    "meteor": "METEOR",
    "spice": "SPICE",
}
PERCENT_METRICS = {"test_acc", "aa_robust_acc", "apgd_ce_subset_acc", "pgd_subset_acc", "corruption_mean_acc", "aulc_test_acc", "aulc_apgd_acc", "ctrl_val_acc", "test_top5", "aa_clean_acc", "t2a_r1", "t2a_r5", "t2a_r10", "a2t_r1", "a2t_r5", "a2t_r10"}


def method_label(m: str) -> str:
    if m in METHOD_STYLE:
        return METHOD_STYLE[m]["label"]
    if m.startswith("caenl_fixed_dcr_l"):
        return f"CAENL + fixed DCR ($\\lambda$={m.split('_l')[-1]})"
    return m.replace("_", " ")


def method_style(m: str) -> dict:
    if m in METHOD_STYLE:
        return METHOD_STYLE[m]
    if m.startswith("caenl_fixed_dcr_l"):
        return {"color": PALETTE["blue"], "ls": ":", "label": method_label(m)}
    return {"color": PALETTE["gray"], "ls": "-", "label": method_label(m)}


def method_sort_key(m: str) -> tuple[int, str]:
    return (METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER), m)


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#9a9995",
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt
