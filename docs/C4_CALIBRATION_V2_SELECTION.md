# C4 calibration-v2 selection record

The archive `caenl-c4-calibration-v2-20260908T033251Z.tar.gz` was audited before
freezing this confirmatory plan. Its external SHA-256 is
`e1a3576d7efa2776168152223dae45838470c749970f86cdbea4cd83ef76a295`;
all 177 retained internal files passed their recorded SHA-256 checks.

Calibration used paired seeds 201 and 202, 9,994,240 training tokens per run,
and 434,176 final held-out evaluation tokens. The selected fixed-DCR condition
was rank 16/4 with lambda 0.01 because it had the best mean validation NLL,
perplexity, and token ECE among the tested fixed settings and improved all three
metrics against the paired baseline in both seeds. The effect was small, so it
is treated only as a calibration signal.

The corrected MACC-Lite configuration produced 48 updates per seed without
saturating at its upper lambda bound. Full MACC produced 48 controller records
and 44/42 learned policy updates; it remains in the confirmatory comparison even
though its 10M-token task-performance effect was mixed. No calibration result is
used as confirmatory evidence.

Frozen confirmatory choices:

- GPT-2 pretrained on C4, sequence length 1024.
- 100M training tokens per run; 2M validation tokens prepared.
- Six paired seeds 301--306.
- Baseline, fixed DCR, MACC-Lite, and Full MACC.
- DCR target ranks h6=16 and h12=4.
- Fixed DCR lambda 0.01.
- MACC-Lite initial lambda 0.005, eta 0.002, bounds [0, 0.08].
- Full MACC settings unchanged from calibration-v2.
- Batch tokens 8192, DCR token subsample 2048, monitoring cadence 25.
