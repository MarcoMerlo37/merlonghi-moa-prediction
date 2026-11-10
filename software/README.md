# Software — MoA Prediction

## 1. Project description

This folder contains the full, runnable solution for the Kaggle **Mechanisms of Action (MoA) Prediction** challenge. The core is two Jupyter notebooks, training and inference; seven further notebooks document how the hyperparameters and the cross-validation protocol were investigated.

| Notebook | Purpose | PDF export | Section |
|---|---|---|---|
| [`moa-training.ipynb`](moa-training.ipynb) | **Training**: exploratory analysis, compound-aware cross-validation of three base models (`CV_SCHEME` = `hybrid`, `grouped` or `row`), nested-CV blending, saving of all trained artifacts and of the OOF predictions, per-target and CV-protocol diagnostics | [`moa-training.pdf`](moa-training.pdf) | 5–7 |
| [`moa-inference.ipynb`](moa-inference.ipynb) | **Inference**: loads the saved artifacts and writes the Kaggle `submission.csv`, with no training | [`moa-inference.pdf`](moa-inference.pdf) | 8 |
| [`moa-tuning.ipynb`](moa-tuning.ipynb) | **Tuning v1**: single-stage Optuna search on 5 row-wise folds | – | 10 |
| [`moa-tuning-row.ipynb`](moa-tuning-row.ipynb) | **Tuning v2**: two-stage search and confirmation on row-wise folds | – | 10 |
| [`moa-tuning-hybrid.ipynb`](moa-tuning-hybrid.ipynb) | **Tuning v3**: v2 with compound-aware folds and paired-seed confirmation | – | 10 |
| [`moa-fold-diagnostics.ipynb`](moa-fold-diagnostics.ipynb) | Fold, compound and protocol diagnostics of a saved run | – | 11 |
| [`moa-blend-offline.ipynb`](moa-blend-offline.ipynb) | Offline recombination of the OOF predictions of two runs | – | 11 |
| [`moa-test-drug-overlap-nn.ipynb`](moa-test-drug-overlap-nn.ipynb) | Indirect analysis of the compound overlap between train and test | – | 11 |
| [`moa-knn-override-freq-drugs.ipynb`](moa-knn-override-freq-drugs.ipynb) | Offline test of a nearest-neighbour override for frequent compounds | – | 11 |

The training notebook builds and cross-validates three base models (TabularResNet, FT-Transformer, CatBoost) on gene-expression / cell-viability profiles and blends them with a nested-CV-validated weighted ensemble. The inference notebook applies the trained models to the test set. The two are separate because the challenge is a **code competition**: a notebook is submitted and re-run by Kaggle on the hidden test set, without internet access. Retraining (75–95 minutes on 2 GPUs for the compound-aware runs) is not needed for that. See the top-level [`README.md`](../README.md) for objectives, methodology and results; the slides are in [`../doc/`](../doc/).

**Order of use.** The notebooks do not have to be run in this order, but the dependencies are:

```
tuning (v2 or v3)  ──►  best_hyperparameters*.json  ──►  moa-training  ──►  artifacts + OOF files
                                                                              │
                                   ┌──────────────────────────────────────────┤
                                   ▼                                          ▼
              moa-fold-diagnostics, moa-blend-offline,              moa-inference  ──►  submission.csv
              moa-knn-override-freq-drugs  (read the OOF files)
              moa-test-drug-overlap-nn     (reads only the competition data)
```

The first cell of each notebook contains a workflow overview, and every major code section of the training and inference notebooks is commented in English and tagged with the matching step number (`# Step N.k`). The notebooks have independent step numbering.

**Kaggle challenge reference:** <https://www.kaggle.com/competitions/lish-moa>

## 2. Environment the notebooks were developed and run in

All notebooks were authored and executed **on Kaggle**, inside Kaggle Notebooks (kernels), not on a local machine. A few things in the code and in the results are tied to that platform:

- **Accelerator:** GPU T4 x2 (two NVIDIA Tesla T4 GPUs) for the training notebook. This is why the code contains GPU-count-aware logic (`N_GPUS`, round-robin device assignment across workers) and comments about the GPUs' VRAM limits. Tuning v1 runs two Optuna jobs in parallel when two GPUs are visible; tuning v2 and v3 run sequentially on the first GPU (`cuda:0`). The diagnostics and analysis notebooks (section 11) need no GPU.
- **Python / image:** Kaggle's standard `python` Docker image (with `numpy`, `pandas`, `torch`, `scikit-learn`, etc. pre-installed). The training runs used Python 3.12; the inference and diagnostics runs, Python 3.13.
- **Data mount:** the competition data is attached as a Kaggle competition source and appears already extracted under `/kaggle/input/competitions/lish-moa`. This path depends on the platform version: if it differs, check `/kaggle/input` in a cell and edit `DATA_DIR` in the training notebook. The inference, tuning-v3 and analysis notebooks find their input folders by file name (or accept an explicit path) and need no edit unless they report an ambiguity.
- **Working directory:** intermediate artefacts (model checkpoints, fold preprocessing pipelines, experiment logs, OOF predictions, the final `submission.csv`) are written to `/kaggle/working`, which Kaggle persists as the notebook's output.
- **Runtime (training):** the compound-aware runs took **75.1 minutes** (run H1: 55.7 min neural-network CV, 16.7 min CatBoost CV, 0.5 min test-set inference) and **95.0 minutes** (run H2: 75.2 and 16.8 min) wall-clock on the two T4s; the earlier row-wise run with default hyperparameters took about **130 minutes** (≈97 min neural-network CV, ≈29 min CatBoost CV, <1 min test-set inference). Timing varies with the platform machine: expect roughly ±20%. The differences between the runs are not explained here. The runtime of the tuning notebooks was not recorded in this repository; tuning v2 and v3 write `elapsed_minutes` to `tuning_report.json`.
- **Internet access:** the training notebook has one network call, `!pip install -q iterative-stratification`, in its first code cell, so the "Internet" toggle must be **on** for that session. The same holds for tuning v1 (`iterative-stratification`), tuning v2 and v3 (`iterative-stratification`, `optuna`, `catboost`) and the diagnostics notebook (`iterative-stratification`). `moa-blend-offline`, `moa-test-drug-overlap-nn`, `moa-knn-override-freq-drugs` and the inference notebook make no network call and run with internet **off**.

Running **on Kaggle** (Option A below) is the path closest to how the notebooks were built and validated. The training notebook can also be run **locally** (Option B): only `DATA_DIR`, `WORK_DIR` and the hyperparameters file need to point elsewhere, and a GPU must be provided by the local machine.

## 3. Data

The raw competition data is **not included in this repository** (over 100 MB, and redistribution outside Kaggle's platform is against the competition rules).

- Data page (permanent link): <https://www.kaggle.com/competitions/lish-moa/data>
- Expected files: `train_features.csv`, `train_targets_scored.csv`, `test_features.csv`, `sample_submission.csv`, and `train_drug.csv` (the mapping from every training signature to its compound, `drug_id`, published by the competition hosts; it is needed by every protocol except `row`, by tuning v3 and by the analysis notebooks, and it has no counterpart for the test set).

The trained artifacts used by the inference notebook are not included either; they are produced by the training notebook (see section 8.2). The same holds for the hyperparameter files (section 10) and the OOF files (section 11).

## 4. Prerequisites (both options)

- Python 3.12 (training) and 3.13 (inference).
- A CUDA-capable GPU is strongly recommended for training and tuning. Without one, training time for the full multi-seed, multi-model cross-validation would be prohibitively long.
- Packages (see [`requirements.txt`](requirements.txt)): `numpy`, `pandas`, `scipy`, `joblib`, `scikit-learn`, `torch`, `iterative-stratification`, `catboost`, `matplotlib` (exploratory plots), and `optuna` (tuning notebooks only). `requirements.txt` lists the packages without versions; [`requirements-frozen.txt`](requirements-frozen.txt) pins the versions of the Kaggle environment of the reported run (written by Step 14.9 of the training notebook; Python itself is not included: 3.12 for training). For a faithful rerun, install from the frozen file. `scikit-learn` and `catboost` matter most: the fold pipelines are `scikit-learn` pickles and the CatBoost models are saved in its own format (reported run: `scikit-learn==1.6.1`, `catboost==1.2.10`). `optuna` is not part of the training environment and is not pinned by the frozen file.
- The inference notebook needs the same libraries, minus `iterative-stratification`. The fold preprocessing pipelines are `scikit-learn` pickles, so the installed `scikit-learn` version should match the one used for training. Step 9.1 of the inference notebook prints the installed versions.

---

## 5. Option A — Run the training notebook on Kaggle (environment it was built in)

1. Go to the competition page, <https://www.kaggle.com/competitions/lish-moa>, and join the competition if required (accepting its rules is needed to access the data).
2. On Kaggle, click **Create → New Notebook**, then **File → Import Notebook** and upload [`moa-training.ipynb`](moa-training.ipynb).
3. Attach the competition data: in the right-hand panel, **Add Input** → search `lish-moa` → **Mechanisms of Action (MoA) Prediction** → **Add**. This mounts the CSV files, including `train_drug.csv`, under `/kaggle/input/competitions/lish-moa`, matching the default `DATA_DIR`. If `train_drug.csv` is not in that folder, the notebook looks for it by name under `/kaggle/input` and stops if it finds none or several.
4. Attach the hyperparameters, or train with the defaults. Upload the JSON file of the tuning notebooks (section 10) as a Dataset, add it as an input, and set `HYPERPARAMETERS_FILE` in Step 1.3 to its path; the default value is a path in the author's account and must be edited. To train with the default hyperparameters instead, set `USE_TUNED_HYPERPARAMETERS = False`.
5. Choose the protocol and the other options in the configuration cell (Step 1.3):

   | Constant | Meaning | Reported runs |
   |---|---|---|
   | `CV_SCHEME` | `"hybrid"`, `"grouped"` or `"row"` | `"hybrid"` (H1, H2), `"row"` (R) |
   | `HYBRID_MIN_ROWS` | used by `"hybrid"`: compounds with more rows than this are split by row | 18 |
   | `USE_TUNED_HYPERPARAMETERS`, `HYPERPARAMETERS_FILE` | read the JSON file, or use the defaults | `True` for H1 and H2 |
   | `HYPERPARAMETER_OVERRIDES` | manual overrides applied after the file, for example `{"FTTransformer": {"batch_size": 64}}` | `{}` |
   | `REFERENCE_OOF_DIR` | Dataset with the `oof_*.npy` and `oof_targets.npy` of a row-wise run, for the per-target comparison in Step 15.5; `None` skips it | a Dataset of run R's OOF |

   Step 1.5 resolves the hyperparameters in three layers (defaults, then the file, then the overrides), validates every value, and prints for each parameter the default, the value used and its source. A key that the notebook does not know is reported and ignored; a value of the wrong type or outside its range stops the run before any training.
6. Set the accelerator: **Notebook Settings** → **Accelerator** → **GPU T4 x2** (any CUDA GPU works, but this reproduces the original setup and runtime; a single-GPU accelerator runs with `N_GPUS = 1`, so fewer jobs run in parallel).
7. Set **Internet: On** in the same settings panel (needed only for the first code cell's `pip install`).
8. Run the notebook top to bottom: **Run → Run All**, or **Restart & Run All**. `submission.csv` is written to `/kaggle/working` and appears in the notebook's **Output** tab once the run finishes.

Besides `submission.csv`, the run writes to `/kaggle/working`:
- everything the inference notebook needs: the 30 network checkpoints, the 10 fold pipelines, the 5 CatBoost models with their metadata, and `metadata.json` (Step 14.9; it also records `cv_scheme`, the architecture arguments of the networks as `model_hyperparameters`, and all hyperparameters used);
- the files used by the analysis notebooks: `oof_resnet.npy`, `oof_ft_transformer.npy`, `oof_catboost.npy`, `oof_targets.npy` (Step 15.3), `cv_fold_of_row.npy`, `train_sig_ids.npy`, `train_drug_ids.npy` (Step 14.9);
- `per_target_report_<model>.csv` (one per model and for the blend, Step 15.4), `per_target_row_vs_cv.csv` (Step 15.5), `experiment_logs/*.json`, `eda_summary.png` and `requirements-frozen.txt`.

To use the artifacts, continue with section 8; for the analysis files, with section 11.

## 6. Option B — Run the training notebook fully locally (not on Kaggle)

### 6.1 Get the code and data

```bash
# from this repository
cd merlonghi-moa-prediction/software

# competition data: download via the Kaggle API (requires a Kaggle account and
# an API token — see https://www.kaggle.com/docs/api#authentication — saved as
# ~/.kaggle/kaggle.json beforehand)
pip install --user kaggle
kaggle competitions download -c lish-moa -p ./data
unzip ./data/lish-moa.zip -d ./data
```

You should now have, under `./data/`: `train_features.csv`, `train_targets_scored.csv`, `test_features.csv`, `sample_submission.csv` and `train_drug.csv`.

### 6.2 Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyterlab             # or `notebook`, to open/run the .ipynb locally
```

`iterative-stratification` is also installed by the notebook's own first code cell, so having it in `requirements.txt` is redundant but harmless.

**GPU users (recommended):** install a `torch` build matching your CUDA version instead of the plain `pip install torch` from `requirements.txt` — see <https://pytorch.org/get-started/locally/> — and install it *before* the rest of `requirements.txt` (or reinstall `torch` afterwards). Without a CUDA-visible GPU the notebook still runs on CPU (`N_GPUS` is detected automatically), just far slower.

### 6.3 Point the notebook at your local paths

Open the notebook (`jupyter lab moa-training.ipynb`) and, in the first configuration code cell (Step 1.3), change:

```python
DATA_DIR = "/kaggle/input/competitions/lish-moa"   # -> "./data" (or wherever you unzipped the CSVs)
WORK_DIR = "/kaggle/working"                       # -> "./working" (or any writable local folder)
HYPERPARAMETERS_FILE = "/kaggle/input/datasets/.../best_hyperparameters_hybrid.json"
                                                   # -> the bare name of a JSON file copied into WORK_DIR (or its path);
                                                   #    or set USE_TUNED_HYPERPARAMETERS = False
REFERENCE_OOF_DIR = "/kaggle/input/datasets/.../moa-reference-oof-row"
                                                   # -> a local folder with the OOF of a row-wise run, or None
```

All paths, including the output `submission.csv`, are derived from these constants. `HYPERPARAMETERS_FILE` is resolved as follows: an existing absolute path is used as it is; a bare file name (for example `"best_hyperparameters_hybrid.json"`) is searched in `WORK_DIR`, then in the current folder, then by name under `/kaggle/input` (exactly one match); an absolute path that does not exist is searched by file name under `/kaggle/input` only. For a local run, the simplest choice is therefore to copy the JSON file into `WORK_DIR` and set `HYPERPARAMETERS_FILE` to its bare name. Also remove or comment out the `!pip install` line if you installed the packages already (it is harmless to keep).

### 6.4 Run

Run all cells top to bottom (`Run → Run All Cells` in Jupyter Lab, or `Kernel → Restart & Run All`). `submission.csv` is written to the `WORK_DIR` you configured above.

**Expected runtime:** on a machine with a single modern CUDA GPU, expect roughly 2–4× the 75–95 minutes of the reference compound-aware runs (the code parallelises across GPUs, so a single-GPU machine loses that 2-way overlap; this factor is an estimate, not a measurement). On CPU only, plan for several hours to a day depending on core count; CatBoost falls back to CPU automatically.

---

## 7. Expected results of the training notebook (to validate a rerun)

A rerun should give values close to the following (out-of-fold mean log loss, treated rows only). Runs H1 and H2 use the same folds (`CV_SCHEME = "hybrid"`, `HYBRID_MIN_ROWS = 18`, seed 42), so a rerun of either can be compared fold by fold; run R used the `row` protocol, default hyperparameters and an earlier revision of the code.

| Model | R (`row`, defaults) | H1 (`hybrid`, tuning v2) | H2 (`hybrid`, tuning v3) |
|---|---|---|---|
| TabularResNet | 0.01632 | 0.01744 | 0.01735 |
| FT-Transformer | 0.01643 | 0.01736 | 0.01744 |
| CatBoost | 0.01699 | 0.01782 | 0.01783 |
| Ensemble (nested) | 0.01607 ± 0.00017 | 0.01714 ± 0.00031 | 0.01715 ± 0.00031 |
| Final blend weights (ResNet / FT-Transformer / CatBoost) | 0.4529 / 0.3746 / 0.1725 | 0.3705 / 0.405 / 0.2246 | 0.478 / 0.3475 / 0.1745 |

Run R can be approximated with `CV_SCHEME = "row"` and `USE_TUNED_HYPERPARAMETERS = False`; this combination was not rerun with the current code, so a small difference from the R column is expected. Runs B and C need the hyperparameter files of tuning v2 and v3 (section 10), whose values are listed in the top-level [`README.md`](../README.md).

The exploratory analysis (Step 2.3) should report 23,814 rows (21,948 treated, 1,866 control), a positive rate of 0.00373 on treated rows, and 53 of 206 targets with fewer than 20 positives. For the compound structure, `describe_drugs` and Step 14.2b (`hybrid`) should report:
- 3,288 distinct compounds in the treated rows (median 6 rows, maximum 718, 111 compounds with at least 10 rows); one identifier among the control rows, shared with no treated row;
- rows held by compounds with more than N rows: 311 compounds / 22.0% for N = 6, 43 / 11.9% for 12, 9 / 9.8% for 18, 8 / 9.7% for 30, 60 and 100;
- "Hybrid groups: 2142 rows of 9 frequent drugs (> 18 rows) split row by row; 3279 other drugs kept whole";
- validation rows per fold `[4386, 4356, 4371, 4383, 4452]`, of which 10.0%, 9.8%, 9.7%, 9.8% and 9.5% have their compound in the training fold;
- targets without positives in the training fold `[4, 4, 4, 4, 3]` and in the validation fold `[36, 36, 36, 35, 32]`.

Mixed precision and parallel GPU jobs make runs non-deterministic: expect differences of the order of 1e-5 in the OOF scores of the row-wise code (the noise of the compound-aware runs was not measured) and a few hundredths in the blend weights, not an exact match.

---

## 8. Inference notebook and Kaggle submission

### 8.1 What it does

[`moa-inference.ipynb`](moa-inference.ipynb) rebuilds the networks, loads the saved weights, fitted preprocessing pipelines and blend weights, and predicts the test set. It reuses the training code for the feature engineering, the architectures, the pipelines (transform only), the inference and the submission checks, with these differences: the pipeline and CatBoost functions read from `ARTIFACT_DIR` instead of `WORK_DIR`; `sanity_check_submission` also accepts a test set without control rows; all training-only code is removed; and the networks are rebuilt from the architecture arguments stored in `metadata.json` (`model_hyperparameters`) instead of the default sizes. The last point is required for tuned models: the ResNet of runs B and C has 768 hidden units, and loading its weights into a 512-unit network fails. The inference notebook must therefore build the model with

```python
def _build_model(model_name, input_dim, output_dim):
    kwargs = metadata.get("model_hyperparameters", {}).get(model_name, {})
    return MODEL_CLASSES[model_name](input_dim=input_dim, output_dim=output_dim, **kwargs)
```

It takes about a minute on a GPU, needs no internet, and writes `/kaggle/working/submission.csv`. It never writes to the artifacts folder, which is read-only on Kaggle.

### 8.2 Create the Dataset of artifacts

1. In the training notebook, run all cells so that Step 14.9 writes `metadata.json` (feature order, column groups, target names, blend weights, folds and seeds, the CV scheme and the model hyperparameters).
2. Save the notebook with **Save Version → Save & Run All (Commit)**.
3. Open the saved version, tab **Output**, and click **New Dataset**.
4. Select these 51 files, name the Dataset (`moa-artifacts` for the reported submissions) and create it:
   - `best_resnet_fold_*_seed*.pth` and `best_ft_transformer_fold_*_seed*.pth` (30 files);
   - `fold_pipeline_*.joblib` and `fold_pipeline_raw_*.joblib` (10 files);
   - `catboost_fold_*.cbm` and `catboost_fold_*_meta.joblib` (10 files);
   - `metadata.json`.

   The OOF files and the other outputs of section 5 are **not** part of this Dataset; if they are saved as a separate Dataset for the analysis notebooks, do not attach it to the inference notebook (see 8.3).

The artifacts used for the reported submissions (runs R and C) are private to the author's account: **[moa-artifacts](https://www.kaggle.com/datasets/marcomerlo37/moa-artifacts)**. They can be regenerated at any time by rerunning the training notebook; each run needs its own set.

### 8.3 Run the inference notebook

1. Create a new Kaggle notebook and import [`moa-inference.ipynb`](moa-inference.ipynb).
2. **Add Input**: the competition `lish-moa` and the Dataset of artifacts. **Attach no other input that contains a `metadata.json`** (for example the old artifacts of another run, or a Dataset with the OOF files of an analysis): the notebook looks for the folder that contains that file and stops if there are several.
3. Set **Internet: Off**. Select a GPU accelerator (a CPU-only run works but the FT-Transformer is slow).
4. Run all cells. The notebook locates the two input folders by file name (`metadata.json` and `test_features.csv`), so no path needs editing. If it does not find exactly one folder for a file, it stops with an explicit message. Step 2.3 compares the installed library versions with those recorded in `metadata.json`. The Kaggle image used for inference may differ from the training one in `python`, `torch` and `numpy` (reported run: Python 3.13 vs. 3.12, `torch` 2.11 vs. 2.10, `numpy` 2.1 vs. 2.0); this is expected and was checked (maximum difference 1.1e-16, section 8.4). A warning is printed only if `scikit-learn` or `catboost` differ, because those two decide whether the pickled pipelines and the CatBoost models load correctly.
5. Step 9.1 checks that the 50 files listed above exist (check_artifacts); metadata.json, the 51st, is read in Step 2.1. The blend weights printed in Step 2.1 must match those of the training run (for run C: ResNet 0.478, FT-Transformer 0.3475, CatBoost 0.1745). Step 9.5 runs the sanity checks and writes `submission.csv`.

### 8.4 Consistency check

On the public test set, the `submission.csv` of the inference notebook was compared with the one produced by the training notebook: for run R the maximum absolute difference is 1.1e-16. This confirms that the artifacts, the feature order and the blend weights are loaded correctly.

The same check should be repeated for every new set of artifacts, before submitting: save the `submission.csv` of the training run as a small Dataset (a file with a unique name, for example `submission_training_H2.csv`, and no `metadata.json`), attach it to the inference notebook and run the last cell (Step 9.6), which is optional.

A maximum difference of the order of 1e-16 is expected; differences up to 1e-9–1e-6 can come from different library versions; larger ones point to an inconsistency (for example artifacts of another run, wrong blend weights, or a `_build_model` that ignores `model_hyperparameters`).

### 8.5 Late submission

The challenge is closed, but a notebook can still be submitted as a **late submission** from the competition page (*Late Submission*), selecting a saved version of the inference notebook. Kaggle re-runs it on the hidden test set and scores the resulting predictions. Late submissions do not appear on the leaderboard.

Save the notebook version first (**Save Version → Save & Run All**): the submission uses the saved version, not the open session. Two submissions were made: the row-wise models (run R), and, after the compound-aware protocol and the tuning were fixed, the hybrid models (run C). Hyperparameters were never tuned on the test score; the first submission did motivate the compound-aware protocols (see the top-level [`README.md`](../README.md)).

### 8.6 Results

| Estimate | Run R | Run C |
|---|---|---|
| Nested-CV blend, treated rows only | 0.01607 ± 0.00017 | 0.01715 ± 0.00031 |
| Nested-CV blend, rescaled to all test rows (× 0.910) | 0.01462 | 0.01561 |
| Kaggle private score | 0.01634 | 0.01643 |
| Kaggle public score | 0.01865 | 0.01873 |

The private score is 11.8% above the rescaled CV estimate for run R and 5.3% for run C, so the row-wise estimate was more optimistic. The difference between the two submissions (+9e-5 private, +8e-5 public) is within the noise of a single submission per configuration. See the top-level [`README.md`](../README.md) for the rescaling and the discussion of the gap.

---


## 9. Ablated models (`ablated-models/`)

`ablated-models/` contains two complete copies of the training notebook, taken at the revision used for the ablation of the main README (row-wise protocol, default hyperparameters, no `train_drug.csv`), with one extra network each: `moa-training-with-MoAMLP.ipynb` (`"mlp"`, three hidden layers 1024-1024-512) and `moa-training-with-MoAConv1D.ipynb` (`"cnn1d"`, dense expansion into a 256×16 pseudo-sequence followed by two 1D convolutions). They are self-contained: run them as they are on Kaggle (section 5, steps 1–3 and 6–8; no hyperparameter file or drug file is needed). They are not used by the delivered training or inference notebooks.
---

## 10. Tuning notebooks

Three versions of the tuning notebook exist. They differ in how a trial is evaluated and in the folds, and the training notebook can use the JSON of any of them. The values that were used, the observations on the searches and the limitations are in the top-level [`README.md`](../README.md#hyperparameter-tuning).

| Notebook | Folds | Trials (ResNet / FT-Transformer / CatBoost) | Evaluation of a trial | Output |
|---|---|---|---|---|
| [`moa-tuning.ipynb`](moa-tuning.ipynb) (v1) | 5 row-wise folds | 30 / 15 / 10 (CatBoost also limited to 1,200 s) | best epoch on the validation fold that is scored (no inner split) | `best_hyperparameters.json` |
| [`moa-tuning-row.ipynb`](moa-tuning-row.ipynb) (v2) | folds 0 and 1 of a row-wise split | 25 / 15 / 15 | 10% row-wise inner split for early stopping; the outer fold is scored once | `best_hyperparameters.json`, `tuning_report.json`, `trials_<model>.csv` |
| [`moa-tuning-hybrid.ipynb`](moa-tuning-hybrid.ipynb) (v3) | folds 0 and 1 of a compound-aware split (`hybrid`, `HYBRID_MIN_ROWS` = 18, own `SPLIT_SEED` = 1234) | 25 / 15 / 15 | 10% compound-aware inner split; confirmation with three paired seeds | `best_hyperparameters_hybrid.json`, `tuning_report.json`, `trials_<model>.csv` |

**What v2 and v3 do.** Stage 1 searches the hyperparameters of each model on fold 0 with Optuna (TPE sampler with `seed=42`, median pruner); the default configuration is enqueued as trial 0, so that it competes in the same study. Stage 2 re-evaluates the best candidates and the default on fold 1 (the top 3 plus the default in v2; the top 2 plus the default, each with the three seeds `CONFIRM_SEEDS`, in v3). A candidate replaces the default only if it passes the adoption rule: in v2, better on both folds with a mean gain of at least `MIN_GAIN` = 5e-5; in v3, a mean gain on the confirmation fold of at least `MIN_GAIN` = 1e-4 **and** a win on more than half of the paired seeds (the search-fold score is reported but not used for the decision, because it is the minimum over the trials). A model for which no candidate passes keeps its default values, which are written to the JSON file. Out-of-memory trials are pruned; other exceptions mark the trial as failed, and the study stops after three consecutive failures. The preprocessing (variance threshold, RankGauss, PCA with 50 and 15 components), the training loop (35 epochs, early-stopping patience 8, label smoothing 0.001) and the CatBoost iterations (250) match the training notebook. `SMOKE_TEST = True` runs a reduced check on a subset of the rows (about 5 minutes according to the v2 notebook) and tags the output files.

**How to run (Kaggle).** Import the notebook, attach the competition data (including `train_drug.csv` for v3), select a GPU accelerator, set **Internet: On** (the first cell installs `optuna`, `catboost` and `iterative-stratification`), and run all cells. Outputs are written to `/kaggle/working`. The notebooks use one GPU (v2 and v3 use `cuda:0` and run the trials sequentially), so the T4 x2 accelerator brings no speed-up for them. The runtime was not recorded; read `elapsed_minutes` in `tuning_report.json` and the `duration` column of `trials_<model>.csv` (Optuna's standard trial table) after a run.

**Using the result in the training notebook.**
1. Upload the JSON file as a Kaggle Dataset (or copy it into `WORK_DIR` for a local run).
2. Point `HYPERPARAMETERS_FILE` to it and keep `USE_TUNED_HYPERPARAMETERS = True` (section 5). Step 1.5 prints a table with the default, the value used and the source of every parameter, so the values that were actually used are always visible in the notebook output.
3. **File names.** The file written by v2 is `best_hyperparameters.json`; it was renamed `best_hyperparameters_hybrid.json` for run H1, although v2 tunes on row-wise folds. The file written by v3 has the same name natively. To avoid mixing the two, archive the output of v2 as `best_hyperparameters_rowtuned.json` and use `best_hyperparameters_hybrid.json` only for the output of v3.

---

## 11. Diagnostics and analysis notebooks

These notebooks work on files saved by a training run, or on the competition data, and need no GPU. For the notebooks that read OOF files, save the following files of the training run as a Dataset (the checkpoints and pipelines are not needed): `oof_resnet.npy`, `oof_ft_transformer.npy`, `oof_catboost.npy`, `oof_targets.npy`, `cv_fold_of_row.npy`, `train_drug_ids.npy`, `metadata.json` (and, optionally, `train_sig_ids.npy`, used as a consistency check). A Dataset with these files also contains a `metadata.json`: it must **not** be attached to the inference notebook (section 8.3).

### 11.1 `moa-fold-diagnostics.ipynb`

Descriptive diagnostics of a compound-aware run, without GPU and without retraining (the notebook's own header states that it runs in a few minutes). It reports: fold sizes, compositions and losses, and whether the frequent compounds explain the outlier fold; the largest compounds, their fold and their targets; the targets dominated by a single compound; a simulation of the `HYBRID_MIN_ROWS` thresholds on the real labels (the two split functions are copied from the training notebook); and the decomposition of the gap between protocols (row, hybrid, grouped) into rows of frequent compounds and all other rows, with a bootstrap over compounds that is paired between runs.

- **Inputs:** `RUN_DIR` (the OOF files above; set it explicitly if several attached Datasets contain `cv_fold_of_row.npy`), `ROW_RUN_DIR` (OOF of a row-wise run, for section 7), optionally `EXTRA_RUN_DIRS` (label → folder). `FREQUENT_MIN_ROWS` must equal the `HYBRID_MIN_ROWS` of the run. The Kaggle scores and `TEST_TREATED_SHARE` used for the back-of-the-envelope inference of section 7 are constants of the notebook.
- **Internet:** on, for `pip install iterative-stratification` (needed by section 6).
- **Output:** printed tables. Reference values for run H0 are in the top-level [`README.md`](../README.md#row-wise-against-compound-aware-cv).

### 11.2 `moa-blend-offline.ipynb`

Compares the nested blend of the OOF predictions of two runs with the same rows and folds: all of the first, all of the second (baseline), ResNet of one run with FT-Transformer and CatBoost of the other, the average of the two runs per model type, and a six-model blend with free weights. A variant is marked as adoptable only if it improves the baseline by at least `MIN_GAIN` (5e-5) and in at least `MIN_FOLDS_BETTER` (4) of the 5 folds, by paired per-fold differences.

- **Inputs:** `H0` (the comparison run) and `H2` (the main run, used as baseline) — two folders with the OOF files above. The notebook asserts that the labels and the fold assignments are identical and stops otherwise. In the reported analysis `H0` was the earlier hybrid run H0 and `H2` run H2.
- **Check:** the baseline must reproduce the nested score printed by the training notebook (0.01715 for run H2); a different value means that the rows are misaligned or that the wrong folder was attached.
- **Output:** a table and the mean blend weights per variant. No variant was adoptable in the reported analysis (best gain 1e-5).

### 11.3 `moa-test-drug-overlap-nn.ipynb`

Indirect analysis of whether the test set shares compounds with the training set (the test set has no compound identifier), from nearest-neighbour cosine distances on the standardised `g-` and `c-` blocks. It builds two reference distributions with pools of the same size as the test pool (validation rows whose compound is in the pool, from random folds; rows whose compound is not, from compound-grouped folds), estimates the share of "seen" rows of the test as a mixture with a bootstrap over compounds, checks the power of the method (AUC between the references and recovery of known shares on pseudo-tests), and analyses each frequent compound (rows assigned to it versus the numbers expected if it were absent or present at its training frequency).

- **Inputs:** only the competition data (`train_features.csv`, `test_features.csv`, `train_drug.csv`); no run output is needed.
- **Outputs:** `nn_distance_ecdf.png`, `frequent_drugs_in_test.csv` and the printed tables.
- **Reading it:** check the AUC first. Below about 0.65 the notebook warns that the estimate of the share is not informative; in the reported analysis it was 0.516, and only the frequent-compound analysis is usable.

### 11.4 `moa-knn-override-freq-drugs.ipynb`

Offline test of a nearest-neighbour override for the rows of frequent compounds. For each hybrid fold, using only the training part, it estimates the distance threshold of each frequent compound (leave-one-out nearest neighbour), its label profile and the precision of the assignment rule; it then assigns the validation rows and compares four override variants (replacement by the profile, mixture weighted by the estimated precision, mixtures with fixed weights 0.5 and 0.9) with the blend, row by row, splitting correct assignments from false positives. It prints the change in OOF loss on all rows, the number of folds improved, the contribution of each compound and a translation into an expected gain on the test set.

- **Inputs:** the OOF files of the run to be tested (`RUN_DIR`, with `train_drug_ids.npy` and `metadata.json` for the blend weights) and the competition data (`train_features.csv`, `train_targets_scored.csv`, `train_drug.csv`). The notebook asserts that the labels and the compound identifiers of the run coincide with the data and stops otherwise.
- **Reading it:** the printed "gain" is the reduction of the OOF loss, so a negative value means that the override **worsens** it. In the reported analysis every variant worsened the loss on all five folds and the override was not adopted.
- **Runtime:** the neighbour search takes about 30 seconds on CPU (printed per fold in Step 3).

---

## 12. Notes

- The delivered notebooks and their PDF exports come from single clean runs (*Restart & Run All*) on Kaggle, so cells are numbered from 1. The delivered training notebook corresponds to run C (compound-aware `hybrid` protocol, hyperparameters of tuning v3).
- The training notebook reproduces the cross-validated scores above. The Kaggle test scores were measured with the inference notebook, once per configuration (section 8.6). The compound-aware protocols use `train_drug.csv`; whether the hidden test set shares compounds with the training set cannot be verified directly, and the indirect analysis of section 11.3 settles it only for the frequent compounds. See the top-level README for the limitations.
- Model checkpoints and fold-level preprocessing artefacts (`*.pth`, `*.joblib`, `*.cbm`) are written to `WORK_DIR` during the training run. They are not part of this repository, are the input of the inference notebook (section 8), and are safe to delete between training runs. The training notebook clears stale fold pipelines at the start of each run. The OOF files (section 5) are overwritten by each run: save them as a Dataset before starting another run if they are needed for comparisons.
