# MarketMirror Submission

This repository contains the code and experiment artifacts for a text-and-structured-feature classification pipeline.

## Overview

The project includes scripts for:

- data preprocessing and feature extraction
- baseline model training and evaluation
- multi-seed reproducibility runs
- concept drift analysis
- statistical comparison and ranking of results
- optional later-stage analysis and visualization

## Repository layout

- `preprocess_only.py` — preprocess raw input data into tensor datasets
- `prepare_phase2_data.py` — build phase 2 training inputs
- `extract_latent_features.py` — extract latent representations from the encoder
- `mm_phase2_expert_trainer.py` — train the expert-gated phase 2 model
- `mm_phase3_physics_engine.py` — derive phase 3 feature statistics
- `mm_phase4_evolution_sim.py` — run evolution/risk simulation experiments
- `mm_phase5_visualizer.py` — generate visual summaries
- `script1_run_baseline_seeds.py` — run baseline experiments across seeds
- `script1b_run_expert_seeds.py` — run expert experiments across seeds
- `script1c_run_core_baselines.py` — evaluate core baseline families
- `script1d_merge_and_rank_results.py` — merge metrics and build rankings
- `script1e_concept_drift_analysis.py` — analyze distribution shift and performance drop
- `script2_build_summary_ci.py` — build summary confidence intervals
- `script3_significance_test.py` — statistical significance testing
- `script4_make_paired_probs.py` — create paired probabilities for comparison
- `pack_project.py` — package the submission into a zip archive

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Typical execution order:

1. preprocess the input data
2. extract latent features
3. build phase 2 training tensors
4. train baseline and expert models
5. merge metrics and run statistical analysis
6. generate plots and summaries

Examples:

```bash
python preprocess_only.py
python extract_latent_features.py
python script1_run_baseline_seeds.py --train-pt ./train2020_phase2_sector.pt --val2022-pt ./val2022_phase2_sector.pt --out-dir ./paper_metrics
python script1d_merge_and_rank_results.py
python script1e_concept_drift_analysis.py --train-pt ./train2020_phase2_sector.pt --val2022-pt ./val2022_phase2_sector.pt --out-dir ./paper_metrics/drift_analysis
```

## Submission notes

- Use relative paths only.
- Avoid embedding personal names, institutions, or local machine paths in code, comments, and logs.
- Keep all experiment outputs inside the repository tree.
- Remove debug prints and local-only configuration before release.
