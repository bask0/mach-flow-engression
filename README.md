# Engression-LSTM for Generative Streamflow Regression

Machine-learning toolbox and experiments for Engression-LSTM paper.

This repository contains code, configuration and notebooks used to train and evaluate temporal models for streamflow regression tasks. It provides utilities to prepare datasets, define models, run experiments and analyze results.

**Key features**
- Reusable PyTorch / PyTorch-Lightning model definitions in `models/`.
- Dataset loaders and harmonization tools in `dataset/`.
- Training and experiment scripts with configurable YAML files in `cli/`.
- Analysis notebooks in `analysis/` for hyperparameter evaluation, CDFs and time-series visualizations.

**Status**: Research/experimental — use for reproduction and development.

## Repository layout

- `cli/` — command-line entry points and configuration templates (`config.yaml`, `config_timesplit.yaml`, `config_qloss.yaml`).
- `dataset/` — data loaders, harmonization scripts and dataset helpers.
- `models/` — model definitions and base classes.
- `utils/` and `analysis/` — plotting, metrics, notebooks and helper scripts.
- `run_experiment.sh`, `run_experiment_timesplit.sh` — example experiment wrappers.
- `create_env.sh` — helper to create a Python environment for this project.

## Requirements

- Python 3.10
- See `requirements.txt` for primary runtime dependencies; the project uses PyTorch / PyTorch-Lightning. A minimal excerpt:

```
pytorch-lightning >= 1.0.0rc2
torch >= 1.3.0
torchvision >= 0.6.0
```

Install dependencies into a virtual environment:

```bash
bash create_env.sh
```

This approach is a bit outdated but does the job.

Download data from `10.5281/zenodo.17900481`. Note that this dataset was used as it was available from a previous project and it contained additional gauging stations than the wiedely-used [CAMELS-CH dataset](https://essd.copernicus.org/articles/15/5755/2023/). We recommend using CAMELS-CH for future work.

## Quick start

1. Prepare your dataset according to the scripts in `dataset/` (see `dataset/machflowdata.py` and `dataset/harmonize_basin_data.py`).
2. Edit configuration files in `cli/` or create a new YAML config based on the templates.
3. Run an experiment (example):

```bash
./run_experiment_timesplit.sh
```

The experiment scripts call into the CLI entrypoints which load the YAML configuration and launch training/evaluation.

## Notebooks and analysis

Use the notebooks in `analysis/` to create  figures:
- `01_hp_eval.ipynb` — hyperparameter evaluation flow
- `02_cdf_eval.ipynb` — cumulative distribution / error analyses
- `03_time_series.ipynb` — time-series plot
- `04_lag.ipynb` — power spectrum analysis

## Configuration

Configuration is YAML-based and lives in `cli/`. The main config keys control dataset paths, model hyperparameters, training settings, logging and evaluation. Copy and adapt `cli/config.yaml` for your experiments.

## Reproducing experiments

1. Create environment and install requirements.
2. Ensure dataset files are placed where your YAML config expects them.
3. Launch `run_experiment.sh` with a chosen config file.
4. Use analysis notebooks to visualize results, adapt paths as needed.

## Citation

Details follow as soon as the associated paper is published.
