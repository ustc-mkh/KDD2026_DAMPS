# Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain

[](http://kdd.org/) [](https://pytorch.org/) [](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for our KDD 2026 paper: **"Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain"**.

In this work, we propose **DAMPS**, a novel framework that calibrates multimodal representations in the spectral domain to enhance recommendation performance.

## 🏗️ Model Architecture

![DAMPS Framework](Figure/DAMPS.png)

> **Figure 1:** The overall architecture of the proposed DAMPS framework.

-----

## ⚙️ Prerequisites

To reproduce the results, please ensure you have the following environment set up.

**Core Dependencies:**

  * **Python:** 3.8.20
  * **PyTorch:** 2.4.1
  * **Torchvision:** 0.19.1
  * **NumPy:** 1.22.4
  * **SpaCy:** 3.7.2
  * **Scikit-learn:** 1.3.2

**Quick Install:**

```bash
pip install -r requirements.txt
# OR manually:
pip install numpy==1.22.4 spacy==3.7.2 scikit-learn==1.3.2 torch==2.4.1 torchvision==0.19.1
```

-----

## 📂 Datasets

We provide pre-processed datasets including text and image features extracted via Sentence-Transformers and CNNs.

### 1\. Amazon Datasets (Baby, Sports, Clothing, Elec)

Download the datasets from Google Drive:
👉 **[Download Link (Baby/Sports/Elec)](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)**

### 2\. MicroLens (Short-video Recommendation)

An alternative dataset for short-video scenarios:
👉 **[Download Link (MicroLens)](https://drive.google.com/drive/folders/14UyTAh_YyDV8vzXteBJiy9jv8TBDK43w?usp=drive_link)**

### Directory Setup

After downloading, please organize your directory as follows:

```text
.
├── data/
│   ├── baby/
│   ├── sports/
│   └── ...
├── src/
│   ├── main.py
│   ├── MGCN.py
│   ├── DAMPS.py
│   └── ...
├── Figure/
│   └── DAMPS.png
└── README.md
```

-----

## 🚀 Usage

### Training the Model

Navigate to the source directory and run `main.py` using the following arguments:

```bash
cd src/
python main.py -m <model_name> -d <dataset_name> -g <gpu_id>
```

**Arguments:**

  * `-m`: Model name (e.g., `SMORE`, `MGCN`).
  * `-d`: Dataset name (e.g., `baby`, `sports`, `clothing`, `elec`).
  * `-g`: GPU IDs separated by commas (e.g., `0` or `0,1,2,3`).

**Example:**

```bash
python main.py -m MGCN -d baby -g 0
```

### Parallel hyperparameter search

Each listed GPU runs one independent parameter combination at a time. When a trial
finishes, that GPU receives the next combination. A single GPU uses the same search
and checkpoint logic sequentially. GPU IDs are passed directly to
`CUDA_VISIBLE_DEVICES` for each worker; use device IDs available on your host.

```bash
cd src
python main.py -m MGCN -d baby -g 0,1,2,3 --threads_per_worker 4
```

The current MGCN YAML defines 192 combinations. Optional `--epochs 100` and
`--stopping_step 10` override the training limits for every trial. `--cpu` runs
sequentially without a GPU. Run from `src` because the default data path is
`../data/`. Each process loads its own data/model, so RAM usage grows with worker
count. Model initialization is locked per dataset to protect shared graph caches;
training runs in parallel. MGCN graph-cache writes are atomic and device-portable.

Each invocation creates a unique directory under `--save_dir` (default:
`./saved_models`):

```text
MGCN-baby-<timestamp>-<unique-id>/
  config.json            # full search configuration
  search.log             # scheduling and current-best progress
  results.json           # all completed/failed trials, updated as trials finish
  best.json              # global winner's parameters and validation/test metrics
  best.pth               # global winner's best-validation-epoch checkpoint
  trial_0000/
    input.json
    train.log
    worker.log           # console output and failure traceback
    result.json
    best.pth             # this trial's best-validation-epoch checkpoint
  trial_0001/
    ...
```

Within each trial, validation improvement saves a checkpoint immediately. Across
trials, the winner is also chosen by validation (`Recall@20` by default), respecting
whether the metric should be maximized or minimized. Equal scores prefer the
lower trial ID. Test metrics correspond to each trial's best validation epoch;
`best_observed_test_trial_id` in `results.json` reports the highest test score
among those epochs for reference only, and does not select the saved model.

The coordinator publishes `best.pth` after each newly winning trial finishes, so
completed results survive an interruption. All trial checkpoints are retained;
plan disk space for one model per combination. Failed trials are logged and other
trials continue; the command reports failure after the queue finishes. Ctrl-C
stops active workers. Automatic resume is not implemented.

Checkpoints now contain metadata as well as weights (they are not a bare
`state_dict`):

```python
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
# Construct the matching model using checkpoint['config'] and its original dataset.
model.load_state_dict(checkpoint['model_state_dict'])
print(checkpoint['parameters'], checkpoint['epoch'])  # epoch is zero-based
print(checkpoint['valid_result'], checkpoint['test_result'])
```

DAMPS phase statistics are included in the saved state. Checkpoints are for model
restoration/evaluation; they do not contain optimizer state for resuming training.

Run regression tests from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
# Also exercise actual concurrent GPU workers on small synthetic data:
TEST_GPU_IDS=1,2 PYTHONPATH=src python -m unittest discover -s tests -v
```

### Switching Between DAMPS and Vanilla MGCN

MGCN uses DAMPS by default. Add `--vanilla` (or set `use_damps: false` in a JSON
configuration) to omit both construction and application of the DAMPS filter.
The original MGCN feature projections, purifier, graph propagation and fusion
remain active.

To run ordinary MGCN once with the best Baby parameters from trial 156:

```bash
cd src
python main.py -m MGCN -d baby -g 1 --vanilla \
  --config_json configs/vanilla_mgcn_baby_trial156.json \
  --save_dir ./saved_models/vanilla --threads_per_worker 2
```

This configuration copies the source experiment's settings and fixes
`n_ui_layers=4`, `n_layers=2`, `cl_loss=0.01`, `knn_k=10`, and `seed=999`.
Training starts from scratch, uses validation-based early stopping, and saves
its best-validation checkpoint in a separate run directory. The epoch budget
is unchanged; the baseline may reach its best validation score at a different
epoch. The JSON includes the source search and trial ID for traceability.

-----

## 📧 Contact

If you have any questions regarding the code or the paper, please feel free to contact:

**Email:** wmh18872323043@163.com

-----

## 📝 Citation

If you find this repository or our paper useful, please cite:

```bibtex
@inproceedings{damsp2026,
  title={Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain},
  author={Minghui Wang, Tingting Zhang, Yu Li, and Yi Chang},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year={2026}
}
