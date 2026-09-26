# CAENL final implementation and cost evidence
Generated UTC: 2026-09-19T10:45:18.556593+00:00
Read-only collection. No model training or evaluation was executed.

## Recorded C4 job settings

````````
[
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/baseline__seed301/job.json",
    "sha256": "ac2b8bf88b375e4d8b2958b778411396148c1080f3c3d8a9d4f7d1b14907ccd5",
    "job_id": "baseline/seed301",
    "type": "language_c4",
    "seed": 301,
    "method": {
      "regularizer": "none",
      "dcr_lambda": 0.0,
      "name": "baseline",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/baseline__seed302/job.json",
    "sha256": "6a7ff5fea082e4bddb950fb16325c225bde0865fa942b1438e7f5ff80b26d8de",
    "job_id": "baseline/seed302",
    "type": "language_c4",
    "seed": 302,
    "method": {
      "regularizer": "none",
      "dcr_lambda": 0.0,
      "name": "baseline",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/baseline__seed303/job.json",
    "sha256": "897b9d55c058b9b0dca1ccad9fca5e8055f895568dbd37d12d3e40bb247422e7",
    "job_id": "baseline/seed303",
    "type": "language_c4",
    "seed": 303,
    "method": {
      "regularizer": "none",
      "dcr_lambda": 0.0,
      "name": "baseline",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/baseline__seed304/job.json",
    "sha256": "f3b1d8627b2afa1f55e823acfc8ad0a52c8829ada73f274e6d42dcb1808344d5",
    "job_id": "baseline/seed304",
    "type": "language_c4",
    "seed": 304,
    "method": {
      "regularizer": "none",
      "dcr_lambda": 0.0,
      "name": "baseline",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/baseline__seed305/job.json",
    "sha256": "59576854b356b61ccd8f9ff7ec8b980ce27090554f8c6f0640c294d18a2bea95",
    "job_id": "baseline/seed305",
    "type": "language_c4",
    "seed": 305,
    "method": {
      "regularizer": "none",
      "dcr_lambda": 0.0,
      "name": "baseline",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/baseline__seed306/job.json",
    "sha256": "145a8c34d119b8a3d59ae8c1bacffcb446ec24ea4beac86bfc7c9934ee99d7ac",
    "job_id": "baseline/seed306",
    "type": "language_c4",
    "seed": 306,
    "method": {
      "regularizer": "none",
      "dcr_lambda": 0.0,
      "name": "baseline",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/fixed_dcr__seed301/job.json",
    "sha256": "f05a57d98622b770aeb84b080c9bb581103d1d117c9985d3c58c81240ec569c7",
    "job_id": "fixed_dcr/seed301",
    "type": "language_c4",
    "seed": 301,
    "method": {
      "regularizer": "fixed_dcr",
      "dcr_lambda": 0.01,
      "name": "fixed_dcr",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/fixed_dcr__seed302/job.json",
    "sha256": "9eadc5516fc6ac73f94ef653ac4b69e8b139bc440a3c0031f851a7fed609332e",
    "job_id": "fixed_dcr/seed302",
    "type": "language_c4",
    "seed": 302,
    "method": {
      "regularizer": "fixed_dcr",
      "dcr_lambda": 0.01,
      "name": "fixed_dcr",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/fixed_dcr__seed303/job.json",
    "sha256": "dc2b445f5e2e171c852a8dcbc6dc1b70d9b64fc8087a32c6459ff61f5ffd5ea6",
    "job_id": "fixed_dcr/seed303",
    "type": "language_c4",
    "seed": 303,
    "method": {
      "regularizer": "fixed_dcr",
      "dcr_lambda": 0.01,
      "name": "fixed_dcr",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/fixed_dcr__seed304/job.json",
    "sha256": "92c93d733b65203c84c136e60250bd0d5be0820504c131fddf0ed292a5ccf892",
    "job_id": "fixed_dcr/seed304",
    "type": "language_c4",
    "seed": 304,
    "method": {
      "regularizer": "fixed_dcr",
      "dcr_lambda": 0.01,
      "name": "fixed_dcr",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/fixed_dcr__seed305/job.json",
    "sha256": "e7a2a9f04d11039c2272518f88e6eadb7a56e9b2878b3847fe49eea560cd8750",
    "job_id": "fixed_dcr/seed305",
    "type": "language_c4",
    "seed": 305,
    "method": {
      "regularizer": "fixed_dcr",
      "dcr_lambda": 0.01,
      "name": "fixed_dcr",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/fixed_dcr__seed306/job.json",
    "sha256": "80d5633716d444ae468b7c041f3833f6af2103039ed59091e9aa15540725d310",
    "job_id": "fixed_dcr/seed306",
    "type": "language_c4",
    "seed": 306,
    "method": {
      "regularizer": "fixed_dcr",
      "dcr_lambda": 0.01,
      "name": "fixed_dcr",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/full_macc__seed301/job.json",
    "sha256": "f9e10d89130624e83b2f8dbeb7a80f6d838ac498ad834d09a58f06501fab3a83",
    "job_id": "full_macc/seed301",
    "type": "language_c4",
    "seed": 301,
    "method": {
      "regularizer": "full_macc",
      "dcr_lambda": 0.005,
      "name": "full_macc",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/full_macc__seed302/job.json",
    "sha256": "2f0314577852819d70270d333041e3c4791cb19a966a7492a40e4a65f08da1fb",
    "job_id": "full_macc/seed302",
    "type": "language_c4",
    "seed": 302,
    "method": {
      "regularizer": "full_macc",
      "dcr_lambda": 0.005,
      "name": "full_macc",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/full_macc__seed303/job.json",
    "sha256": "6cd02e9401b881d9d5ec545989701a2659e835e7e3e1a30ff4c5a911ab3c3c06",
    "job_id": "full_macc/seed303",
    "type": "language_c4",
    "seed": 303,
    "method": {
      "regularizer": "full_macc",
      "dcr_lambda": 0.005,
      "name": "full_macc",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/full_macc__seed304/job.json",
    "sha256": "e1811c16258d05d6f32ede9d1cbe2599e843b016eaabee2e4e683f6436432f71",
    "job_id": "full_macc/seed304",
    "type": "language_c4",
    "seed": 304,
    "method": {
      "regularizer": "full_macc",
      "dcr_lambda": 0.005,
      "name": "full_macc",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/full_macc__seed305/job.json",
    "sha256": "752cd827c1273674a1bdcf2d388a6262ecf8914df0263778358999f4f6c987ab",
    "job_id": "full_macc/seed305",
    "type": "language_c4",
    "seed": 305,
    "method": {
      "regularizer": "full_macc",
      "dcr_lambda": 0.005,
      "name": "full_macc",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/full_macc__seed306/job.json",
    "sha256": "9a3db4c2b7ef97cdc642337a780a16440362cf84125806b40ef3a4915b1e55cb",
    "job_id": "full_macc/seed306",
    "type": "language_c4",
    "seed": 306,
    "method": {
      "regularizer": "full_macc",
      "dcr_lambda": 0.005,
      "name": "full_macc",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/macc_lite__seed301/job.json",
    "sha256": "822fd6cb0782b1867118da5020a5dcea233114b9b16c7acff1b59e97b5f38b2f",
    "job_id": "macc_lite/seed301",
    "type": "language_c4",
    "seed": 301,
    "method": {
      "regularizer": "macc_lite",
      "dcr_lambda": 0.005,
      "name": "macc_lite",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/macc_lite__seed302/job.json",
    "sha256": "73ddd06708b90af3cae297f799becc256190472202e621d9abc8f2a7f023b653",
    "job_id": "macc_lite/seed302",
    "type": "language_c4",
    "seed": 302,
    "method": {
      "regularizer": "macc_lite",
      "dcr_lambda": 0.005,
      "name": "macc_lite",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/macc_lite__seed303/job.json",
    "sha256": "812dbd0bc9e2686f15bfd265bb76974cca2c9095115babbd176f8568ac3de7c3",
    "job_id": "macc_lite/seed303",
    "type": "language_c4",
    "seed": 303,
    "method": {
      "regularizer": "macc_lite",
      "dcr_lambda": 0.005,
      "name": "macc_lite",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/macc_lite__seed304/job.json",
    "sha256": "655dbfded43bdbaccedc62ec64227e2a555ecb13b1998bf2aec4d929fd8baf65",
    "job_id": "macc_lite/seed304",
    "type": "language_c4",
    "seed": 304,
    "method": {
      "regularizer": "macc_lite",
      "dcr_lambda": 0.005,
      "name": "macc_lite",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/macc_lite__seed305/job.json",
    "sha256": "0d98ef5215ada04b2a3225a794a5cc703d6940d04cc94e7c07fc733ab189eaee",
    "job_id": "macc_lite/seed305",
    "type": "language_c4",
    "seed": 305,
    "method": {
      "regularizer": "macc_lite",
      "dcr_lambda": 0.005,
      "name": "macc_lite",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  },
  {
    "path": "/mnt/caenl/active/results/caenl-c4-confirmatory-v2/stages/c4_confirmatory/jobs/macc_lite__seed306/job.json",
    "sha256": "178de227200c4dc596711c2ce2d0fb401152330795e180c00c31054f222ee58b",
    "job_id": "macc_lite/seed306",
    "type": "language_c4",
    "seed": 306,
    "method": {
      "regularizer": "macc_lite",
      "dcr_lambda": 0.005,
      "name": "macc_lite",
      "acquisition": "random",
      "adversarial_training": false,
      "full_supervision": false
    },
    "language": {
      "dataset": "allenai/c4",
      "subset": "en",
      "train_shards": [
        "en/c4-train.00000-of-01024.json.gz"
      ],
      "validation_shards": [
        "en/c4-validation.00000-of-00008.json.gz"
      ],
      "model": "gpt2",
      "init": "pretrained",
      "sequence_length": 1024,
      "train_tokens": 100000000,
      "validation_tokens": 2000000,
      "batch_tokens": 8192,
      "lr": 5e-05,
      "weight_decay": 0.01,
      "warmup_steps": 200,
      "monitored_layers": [
        6,
        12
      ],
      "dcr_token_subsample": 2048,
      "configured_rank": {
        "h6": 16,
        "h12": 4
      },
      "controller_error_mode": "absolute_log_ratio",
      "target_effective_rank": {
        "h6": 16,
        "h12": 4
      },
      "monitoring_cadence_steps": 25,
      "eval_every_steps": 500,
      "metrics": [
        "validation_nll",
        "perplexity",
        "token_ece"
      ],
      "ctrl_val_sequences": 64,
      "checkpoint_every_steps": 500,
      "log_every_steps": 50
    },
    "reproducibility": {
      "deterministic_algorithms": false,
      "cudnn_benchmark": true,
      "save_pip_freeze": true,
      "telemetry_interval_s": 10,
      "max_device_data_fraction": 0.45
    }
  }
]
````````


## C4 archive identity

````````
{
  "path": "/mnt/caenl/persistent/archives/caenl-c4-confirmatory-v2-20260908T221848Z.tar.gz",
  "expected_sha256": "e87436f3e57ccd6a152fc32bd859a24859c07382a3a8ff7167db1402dff76f8b",
  "actual_sha256": "e87436f3e57ccd6a152fc32bd859a24859c07382a3a8ff7167db1402dff76f8b",
  "matched": true
}
````````


