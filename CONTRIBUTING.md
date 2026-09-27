# Repository scope and contributions

This repository contains research software and the materials needed to execute and analyze its experiments:

- Source code, configurations, dependency specifications, tests, and run instructions.
- Software validation records and source/protocol/environment provenance.
- Non-sensitive numerical exports, with their identities and verification scope.
- Plotting code and numeric inputs; generated plots can be rebuilt locally.

Keep the article and its publication process outside this repository. Do not add manuscript PDFs, LaTeX or Word article sources, journal class/style templates, submission bundles, decision letters, reviewer correspondence, or author-facing editorial worklists. Do not package these files inside a code archive. The ignore rules help avoid accidental additions; inspect staged files before committing.

Keep licensed datasets, reference captions, credentials, private account information, and large model states outside the public repository. Preserve applicable third-party notices; adding code here does not create or change its license.

## Preserve running experiments

The feedback study in progress uses the launcher and engine frozen at commit `65df044feaf3bb6604416b1011ede5ea3b2a36ed`. Preserve that release's code, checksums, protocol, seed sets, and statistical families. Documentation can be improved without changing or restarting the server process.

If an execution bug requires a code change, record it as a separately identified revision with its validation and a clear account of affected runs. Do not overwrite existing results, silently substitute seeds, or change settings in response to evaluation outcomes.

## Result updates

Publish only reviewed numeric exports and their provenance. State whether a validation record concerns software tests, exported-record arithmetic, saved-model inference, or a new training run. Mark partial campaigns as incomplete and retain negative findings. Keep new feedback-study results separate from the earlier `results/2026-09-26/` snapshot.
