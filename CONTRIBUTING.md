# Contributing

Contributions should improve executable research code, experiment definitions, tests, or documentation that enables independent reproduction.

## Documentation

Write instructions for readers who have no access to the original development environment or correspondence. State prerequisites, supported hardware, data requirements, commands, outputs, and known limitations. Use configurable SSH destinations rather than personal hostnames or account details. Keep operational status messages and editorial preparation notes outside the documentation.

## Versioned experiments

Identify each experiment by its source revision, resolved configuration, dataset identities, seed sets, and environment. Keep these fixed for a run and its resumed checkpoints. Release execution changes under a distinct revision and describe their effect on reproducibility. Do not overwrite existing outputs or substitute seeds in response to evaluation outcomes.

The reference feedback implementation is identified by commit `65df044feaf3bb6604416b1011ede5ea3b2a36ed`. Its engine manifest and launcher checksum permit verification independently of documentation revisions.

## Numerical records

Include checksums and provenance with result exports. Distinguish software tests, exported-record arithmetic, saved-model inference, and new training runs. Label incomplete matrices accurately, retain negative findings, and keep separate experiments in distinct result directories.

## Repository content

Include source code, configurations, run instructions, tests, non-sensitive numeric records, and plotting code. Generate graphics locally when needed. Article files, journal templates, submission correspondence, private worklists, credentials, licensed datasets, and large checkpoints are outside the repository scope. Check archives as well as individual staged files before adding them.

Preserve third-party notices and existing licensing terms. Cite this software and the exact commit used through the metadata in `CITATION.cff`.
