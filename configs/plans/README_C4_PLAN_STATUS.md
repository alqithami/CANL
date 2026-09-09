# C4 plan status in v5.4.3

Calibration-v2 has been audited. The selected fixed-DCR setting is the
rank-16/4 target with `dcr_lambda: 0.01`; the corrected MACC-Lite and Full MACC
settings are retained unchanged for confirmatory evaluation.

Run only:

```text
configs/plans/one_c4_confirmatory_v2.yaml
```

It executes six paired seeds (301--306) for `baseline`, `fixed_dcr`,
`macc_lite`, and `full_macc`, using 100M C4 training tokens per run. The old
pilot-v1, confirmatory-v1, calibration-v2, and pre-calibration template are
retained only for provenance and are blocked by the one-dataset launcher.
