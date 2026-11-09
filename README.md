# MoA Prediction — Multi-Model Ensemble

Author: Marco Merlonghi — Project Work in Machine Learning and Data Mining

## Overview

This project addresses the Kaggle **Mechanisms of Action (MoA) Prediction** challenge: given gene-expression and cell-viability profiles of human cells treated with a chemical compound, predict which of 206 possible biological mechanisms of action (MoAs) the compound triggers. It is a **multi-label classification** problem (a sample can be positive for zero, one, or several of the 206 targets simultaneously) evaluated with the mean binary log loss over all samples and targets [1].

- Competition page: <https://www.kaggle.com/competitions/lish-moa>
- Data: <https://www.kaggle.com/competitions/lish-moa/data>

The solution is delivered as a **training** notebook, which cross-validates the models and saves them, and an **inference** notebook, which loads the saved models and produces the Kaggle submission (see [`software/`](merlonghi-moa-prediction/software/)). Around them sit three **tuning** notebooks, a **fold and compound diagnostics** notebook and three **offline analysis** notebooks, which document how the cross-validation protocol and the hyperparameters were investigated (see [Notebooks](#notebooks)). Two models were submitted to the Kaggle test set as late submissions: the first was trained with row-wise cross-validation and default hyperparameters (run R), the second with the compound-aware hybrid cross-validation and tuned hyperparameters (run H2) (see [Results](#results)).

### Data in brief

Figures from the exploratory analysis in the training notebook (Step 2.3 and `describe_drugs`):
- 23,814 training rows: 21,948 treated and 1,866 control rows (7.8%, `cp_type == ctl_vehicle`). Controls have no MoA by definition: none of them has a positive label. They are excluded from training and set to 0 at submission.
- 772 gene-expression features (`g-`), 100 cell-viability features (`c-`), plus the categorical `cp_time` (24, 48, 72 h) and `cp_dose` (D1, D2).
- Positives are very rare: 0.373% of the (treated row, target) cells are positive. 34.2% of the treated rows have no label, 57.1% have one, 8.7% have two or more (at most 7).
- The 206 targets are highly imbalanced: positives per target range from 1 to 832 (median 38), and 53 targets have fewer than 20 positives.
- `train_drug.csv`, published by the competition hosts, maps every training signature to a compound (`drug_id`). The treated rows contain 3,288 distinct compounds, with a median of 6 rows and a maximum of 718 rows per compound. The 9 compounds with more than 18 rows hold 9.8% of the treated rows. The control rows share a single identifier that no treated row has. **The test set has no compound identifier.**

## Objectives

- Build a robust cross-validated pipeline for a large-scale, highly imbalanced multi-label classification problem.
- Compare several modelling paradigms on the same folds: a residual tabular network (TabularResNet), an attention-based FT-Transformer [2], and a gradient-boosted tree model (CatBoost) [3], each fed a preprocessing pipeline suited to it.
- Reduce variance through multi-seed bagging and 5-fold multilabel-stratified cross validation [4, 5], and combine the base models with an ensemble whose weights are learned and re-validated through nested cross-validation.
- Make the cross-validation compound-aware: use `train_drug.csv` to build folds that keep a compound in a single fold (three protocols: `row`, `grouped`, `hybrid`), and quantify how optimistic the row-wise estimate was.
- Tune the hyperparameters of the three models with Optuna [12, 13] under a protocol that never uses the outer validation fold for model selection, and test whether tuning matters under the compound-aware cross-validation.
- Check the cross-validated estimate against an independent measure: the Kaggle test set (two late submissions; see [Results](#independent-check-on-the-kaggle-test-set) for how it was and was not used).
- Produce a clean, reproducible, well-documented deliverable (see [`software/`](merlonghi-moa-prediction/software/)).

## Notebooks

All notebooks are in [`software/`](merlonghi-moa-prediction/software/); setup and run instructions are in [`software/README.md`](merlonghi-moa-prediction/software/README.md).

| Notebook | Role | Main outputs |
|---|---|---|
| [`moa-training.ipynb`](merlonghi-moa-prediction/software/moa-training.ipynb) | Training and cross-validation of ResNet, FT-Transformer and CatBoost under `CV_SCHEME` = `hybrid` (default), `grouped` or `row`; hyperparameters read from a JSON file (Step 1.5); nested-CV blend; per-target error analysis (Step 15.4) and CV-protocol diagnostics (Step 15.5) | 51 artifacts for inference, `submission.csv`, OOF predictions (`oof_*.npy`), fold and compound bookkeeping, per-target reports |
| [`moa-inference.ipynb`](merlonghi-moa-prediction/software/moa-inference.ipynb) | Loads the saved artifacts and writes the Kaggle submission, no training; rebuilds the networks from `model_hyperparameters` in `metadata.json` | `submission.csv` |
| [`moa-tuning.ipynb`](merlonghi-moa-prediction/software/moa-tuning.ipynb) | Tuning v1: single-stage Optuna search (30/15/10 trials for ResNet/FT-Transformer/CatBoost) on 5 row-wise folds; the objective is the mean over folds of the best validation loss measured on the validation fold itself | `best_hyperparameters.json` |
| [`moa-tuning-row.ipynb`](merlonghi-moa-prediction/software/moa-tuning-row.ipynb) | Tuning v2: two stages on row-wise folds. Search on fold 0 (25/15/15 trials, 10% inner split for early stopping, default configuration enqueued as trial 0), confirmation of the top 3 candidates and of the default on fold 1; a tuned model is adopted only if it beats the default on both folds by at least 5e-5 | `best_hyperparameters.json`, `tuning_report.json`, `trials_*.csv` |
| [`moa-tuning-hybrid.ipynb`](merlonghi-moa-prediction/software/moa-tuning-hybrid.ipynb) | Tuning v3: v2 with compound-aware folds (outer folds and inner split built like the training notebook's hybrid protocol, own split seed), wider search spaces, a longer pruner warm-up, and a confirmation on fold 1 with three paired seeds. A candidate is adopted only if the mean gain on the confirmation fold is at least 1e-4 **and** it wins on more than half of the seeds | `best_hyperparameters_hybrid.json`, `tuning_report.json`, `trials_*.csv` |
| [`moa-fold-diagnostics.ipynb`](merlonghi-moa-prediction/software/moa-fold-diagnostics.ipynb) | CPU-only diagnostics of a compound-aware run from its saved OOF files: fold sizes and losses, the largest compounds, targets dominated by a single compound, simulation of the `HYBRID_MIN_ROWS` threshold, and decomposition of the gap between protocols by compound frequency (with a bootstrap over compounds) | printed tables |
| [`moa-blend-offline.ipynb`](merlonghi-moa-prediction/software/moa-blend-offline.ipynb) | Offline recombination of the OOF predictions of two runs with identical rows and folds (mixing models, averaging, six-model blend), scored with nested blending and paired per-fold differences | printed table |
| [`moa-test-drug-overlap-nn.ipynb`](merlonghi-moa-prediction/software/moa-test-drug-overlap-nn.ipynb) | Indirect check of whether the test set shares compounds with the training set, from nearest-neighbour distances in feature space, with a power check (AUC and recovery of a known mixture) and a per-compound analysis of the frequent compounds | `nn_distance_ecdf.png`, `frequent_drugs_in_test.csv` |
| [`moa-knn-override-freq-drugs.ipynb`](merlonghi-moa-prediction/software/moa-knn-override-freq-drugs.ipynb) | Offline test, on the hybrid OOF, of a nearest-neighbour override of the predictions for the rows of frequent compounds | printed tables |

### Runs referred to in the results

| Run | CV protocol | Hyperparameters | Kaggle late submission |
|---|---|---|---|
| R | `row` (earlier revision of the code) | defaults | yes (private 0.01634, public 0.01865) |
| H1 | `hybrid` | tuning v2 (row-wise folds) | no |
| H2 | `hybrid` | tuning v3 (compound-aware folds) | yes (private 0.01643, public 0.01873) |

The diagnostics notebook was applied to a further, earlier `hybrid` run (called H0 below; nested-blend OOF 0.01720). Its hyperparameters are not documented in this repository, so H0 is used only where explicitly mentioned.

## Approach summary

| Stage | Choice |
|---|---|
| Cross-validation | 5-fold multilabel-stratified (`iterative-stratification`), `random_state=42`, on treated rows only. Three protocols selected by `CV_SCHEME`: `"hybrid"` (default), `"grouped"`, `"row"` (see [Cross-validation protocols](#cross-validation-protocols)) |
| Model selection (neural nets) | Inner split (10% of each training fold) for early stopping, LR scheduling and best-epoch choice, built with the same grouping rule as the outer folds; the outer fold is scored once |
| Hyperparameter tuning | Optuna (TPE sampler, median pruner), three notebook versions (see [Hyperparameter tuning](#hyperparameter-tuning)); the values enter the training notebook through a JSON file validated in Step 1.5 |
| Preprocessing (ResNet) | `QuantileTransformer` (RankGauss) → PCA (default: 50 components on gene features, 15 on cell-viability features), concatenated to the transformed features [11] |
| Preprocessing (FT-Transformer, CatBoost) | `QuantileTransformer` (RankGauss), no PCA [11] |
| Base models | TabularResNet, FT-Transformer (3-seed bagging each), CatBoost (`MultiLogloss`, fixed 250 iterations) |
| Constant targets (CatBoost) | A target with no positive in a training fold is excluded from the fit and predicted with a smoothed prior (Jeffreys, 0.5 pseudo-counts) instead of 0 |
| Loss | `BCEWithLogitsLoss` [10], label smoothing 0.001 [8] |
| Optimisation | AdamW [6], `ReduceLROnPlateau`, mixed precision [7] |
| Regularisation | Early stopping (patience 8), dropout, batch norm, weight decay |
| Ensembling | Softmax-reparameterised weight search (Nelder–Mead [9]) on out-of-fold predictions, re-validated with nested CV |
| Post-processing | Control-vehicle test rows set to 0. A nearest-neighbour override for frequent compounds was tested offline and rejected |
| Diagnostics | Per-target error analysis (skill against the per-target prior, calibration, loss share) with bootstrap over compounds; CV-protocol diagnostics; fold and compound diagnostics; indirect test-compound-overlap analysis |
| Inference | Separate notebook: loads the 30 network checkpoints, 5 CatBoost models, 10 fold pipelines and the blend weights; no training |

Architectural and implementation details are documented inline in the notebooks — see [`software/moa-training.ipynb`](merlonghi-moa-prediction/software/moa-training.ipynb) (training) and [`software/moa-inference.ipynb`](merlonghi-moa-prediction/software/moa-inference.ipynb) (inference). The FT-Transformer is a simplified variant of the one in [2] (post-LayerNorm encoder, GELU feed-forward, no ReGLU). The slides are in [`doc/`](merlonghi-moa-prediction/doc/).

### Cross-validation protocols

The competition hosts published `train_drug.csv`, so the earlier assumption that the compound cannot be controlled in the cross-validation no longer holds for the training set. The protocol is selected with `CV_SCHEME` (training notebook Steps 1.3 and 2.5):

| Protocol | Unit kept in one fold | Properties |
|---|---|---|
| `row` | none (every row is independent) | The earlier protocol. 99.4% of the validation rows have their compound in the training fold |
| `grouped` | the compound | Stratification at compound level (a compound is positive for a target if any of its rows is). Fold sizes are unbalanced (largest deviation from the mean fold size 25.6% with seed 42), because compounds have up to 718 rows |
| `hybrid` (default) | the compound, except that the rows of compounds with more than `HYBRID_MIN_ROWS` = 18 rows are split row by row | 9 compounds, 9.8% of the treated rows, are split by row; fold sizes within 1.4% of the mean; about 9.8% of the validation rows share their compound with the training fold |

The inner split used for early stopping follows the same rule. `HYBRID_MIN_ROWS` is a design choice. The diagnostics notebook simulates the candidate thresholds on the real labels (one splitter seed):

| Scheme | Rows split by row | Max fold-size deviation | Validation rows with seen compound | Targets without positives in validation (mean) |
|---|---|---|---|---|
| pure grouped | 0% | 25.6% | 0% | 35.2 |
| hybrid, > 6 rows | 22.0% | 3.4% | 22.0% | 29.8 |
| hybrid, > 12 rows | 11.9% | 3.1% | 11.9% | 34.6 |
| hybrid, > 18 rows | 9.8% | 1.4% | 9.8% | 35.0 |
| hybrid, > 30, 60 or 100 rows | 9.7% | 2.8% | 9.7% | 35.0 |

The thresholds 30, 60 and 100 give identical splits (8 compounds). The threshold 18 is the largest one with the smallest fold-size deviation for this seed; the difference between 1.4% and 2.8% is within what a different seed could change, and the dependence of the CV loss on the threshold was not measured. Under `hybrid`, the targets without positives in the training folds are 3–4 (of 206) and 32–36 in the validation folds.

### Hyperparameter tuning

Three versions of the tuning notebook exist (see [Notebooks](#notebooks)). Their JSON outputs were used as follows:

| Version | Folds | Model selection inside a trial | JSON used by |
|---|---|---|---|
| v1 `moa-tuning` | 5 row-wise | the validation fold itself (best epoch on that fold) | no reported run |
| v2 `moa-tuning-row` | 2 row-wise folds (0 and 1) | 10% row-wise inner split | run H1(file renamed `*_hybrid.json`) |
| v3 `moa-tuning-hybrid` | 2 compound-aware folds (0 and 1, own seed) | 10% compound-aware inner split | run H2 |

The v2 file is named `best_hyperparameters_hybrid.json` in the earlier Kaggle Dataset only because it was renamed after the run; **it was tuned on row-wise folds**. To avoid confusion, the file of v2 should be archived as `best_hyperparameters_rowtuned.json` and `best_hyperparameters_hybrid.json` reserved for the output of v3.

Values used (defaults in the first column; H1 and H2 come from the JSON files, as printed by Step 1.5 of the training notebook):

| Model | Parameter | Default | Run H1 | Run H2 |
|---|---|---|---|---|
| ResNet | learning rate | 1e-3 | 1.18e-3 | 2.67e-3 |
| | dropout | 0.2 | 0.427 | 0.583 |
| | weight decay | 1e-5 | 8.5e-6 | 2.66e-4 |
| | hidden units | 512 | 768 | 768 |
| | batch size | 256 | 128 | 512 |
| FT-Transformer | learning rate | 1e-3 | 2.35e-3 | 1e-3 (default) |
| | dropout | 0.2 | 0.063 | 0.2 (default) |
| | weight decay | 1e-5 | 9.9e-6 | 1e-5 (default) |
| | tokens / blocks / heads / batch | 64 / 2 / 4 / 64 | 64 / 2 / 4 / 64 | 64 / 2 / 4 / 64 |
| CatBoost | depth | 5 | 4 | 4 |
| | `l2_leaf_reg` | 3 | 19.0 | 56.0 |
| | learning rate | 0.08 | 0.1034 | 0.1034 |
| | `subsample` (Bernoulli) | – | 0.829 | 0.829 |
| | iterations | 250 | 250 | 250 |

Observations (they describe the searches, not a measured effect):
- In v3 no FT-Transformer candidate beat the default on the confirmation fold, so the FT-Transformer of run H2 uses the default configuration.
- CatBoost `learning_rate` and `subsample` are identical in H1 and C, which is consistent with the same seeded random draw of the sampler (`TPESampler(seed=42)`). `l2_leaf_reg` sits at the same quantile (0.983 on a log scale) of both search ranges ([1, 20] in v2 and [1, 60] in v3): the search did not locate an interior optimum for it, and the CatBoost OOF does not change (0.01782 in H1, 0.01783 in H2).
- In v3, the ResNet dropout (0.583, range [0.1, 0.6]), batch size (512, the largest value) and learning rate (2.67e-3, range [3e-4, 3e-3]) lie at or near the upper bounds of the search space, so the optimum may lie outside it.

## Results

### Summary of the findings

- Under the compound-aware `hybrid` CV the nested blend scores 0.01715 (run H2) and 0.01714 (run H1) against 0.01607 under the `row` CV (run R): +6.7%. The comparison is not controlled (different hyperparameters and code revision), see below.
- Tuning does not change the hybrid result measurably: the blend of H2 differs from that of H1 by +1e-5, and no recombination of the OOF predictions of two hybrid runs improves on H2 by the 5e-5 threshold.
- The two late submissions score within 1e-4 of each other on the private (0.01634 vs 0.01643) and on the public set (0.01865 vs 0.01873). The hybrid CV estimate is closer to the test score (private +5.3% against +11.8% for the row-wise estimate), not because the models are better.
- The test set contains frequent compounds of the training set: at least five of the eight analysed appear at about their training frequency. For ordinary compounds the overlap cannot be assessed with the proxy used (AUC 0.516).
- A nearest-neighbour override on the frequent compounds worsens the OOF loss and was not adopted.

### Cross-validation

Out-of-fold (OOF) mean log loss over the 5 folds, treated rows only (lower is better). Network scores use predictions averaged over 3 seeds.

| Model | R (`row`, defaults) | H1 (`hybrid`, tuning v2) | H2 (`hybrid`, tuning v3) |
|---|---|---|---|
| Baseline (per-target prior of the training fold) | 0.0222 | 0.02233 | 0.02233 |
| TabularResNet | **0.01632** | 0.01744 | 0.01735 |
| FT-Transformer | 0.01643 | 0.01736 | 0.01744 |
| CatBoost | 0.01699 | 0.01782 | 0.01783 |
| **Ensemble (nested, honest estimate)** | **0.01607 ± 0.00017** | **0.01714 ± 0.00031** | **0.01715 ± 0.00031** |
| Blend weights (ResNet / FT-Transformer / CatBoost) | 0.4529 / 0.3746 / 0.1725 | 0.3705 / 0.405 / 0.2246 | 0.478 / 0.3475 / 0.1745 |
| Total run time on 2× T4 | ≈130 min | 75.1 min | 95.0 min |

Two blend scores are computed. The *nested* score fits the weights on 4 folds and scores them on the fifth, for each fold in turn. The *in-sample* score fits the weights on all OOF rows and scores them on the same rows. The difference measures the optimism of the blending step: 1.58e-6 for R (0.01606 vs 0.01607), 2.84e-6 for H1 and 2.57e-6 for H2. With only three weights to fit it is negligible. This says nothing about leakage from the CV design itself (see [Limitations](#limitations)). The nested score is 27.6% (R) and 23.2% (H2) lower than the baseline.

The standard deviation across folds is larger under the hybrid protocol (0.00031 against 0.00017): the folds are less homogeneous when compounds are kept together.

#### Run R (row-wise, default hyperparameters)

The ensemble is about 1.5% better than the best single model (ResNet). TabularResNet and FT-Transformer carry about 83% of the weight. CatBoost, despite the weakest standalone score, still earns 17% of it. Two measurements support its contribution:

- **Different errors.** The correlation between the models' residuals (OOF prediction minus label, Step 15.3) is lower for CatBoost than between the two networks, especially on the positive cells:

| Model pair | All cells | Positive cells |
|---|---|---|
| ResNet – FT-Transformer | 0.987 | 0.977 |
| ResNet – CatBoost | 0.973 | 0.930 |
| FT-Transformer – CatBoost | 0.972 | 0.928 |

- **Drop-one blends.** Each row is the nested-CV blend of the two remaining models (Step 15.3):

| Model excluded | OOF log loss | Difference vs. full blend (0.01607) | Weights |
|---|---|---|---|
| ResNet | 0.01624 ± 0.00013 | +0.00017 | FT-Transformer 0.6789, CatBoost 0.3211 |
| FT-Transformer | 0.01619 ± 0.00019 | +0.00012 | ResNet 0.714, CatBoost 0.286 |
| CatBoost | 0.01612 ± 0.00018 | +0.00005 | ResNet 0.5474, FT-Transformer 0.4526 |

The gain from CatBoost is small, but it is consistent: removing it worsens the nested score in all five folds (by 6, 7, 3, 1 and 8 × 10⁻⁵, from the per-fold scores printed in Step 14.5 and Step 15.3). The standard deviation across folds is not the right yardstick for this comparison, because the folds are the same for both blends.

Per-fold scores in the notebook output are the mean of the per-seed losses; they are higher than the OOF scores, which use seed-averaged predictions (bagging gain of about 0.0005 per network in run R).

A full training run of R (5 folds × 2 networks × 3 seeds, plus the CatBoost CV) took about 130 minutes on a Kaggle notebook with 2× NVIDIA T4: 128 min for the main pipeline as printed by the notebook (96.6 min neural-network CV, 28.7 min CatBoost CV, 0.5 min test-set inference, the rest being blending and logging) and 131 min for the whole notebook including the final checks. Wall-clock time is noisy on the platform: the ablation configurations below, which train more jobs, finished in 106–113 minutes. The hybrid runs took less (B: 55.7 min neural-network CV, 16.7 min CatBoost CV; C: 75.2 and 16.8 min); the differences between runs are not explained here.

#### Run H2 (hybrid, tuning v3)

The ensemble is 1.2% better than the best single model (ResNet, 0.01735). Residual correlations (Step 15.3) are 0.988 / 0.978 / 0.978 between the model pairs on all cells (ResNet–FT, ResNet–CatBoost, FT–CatBoost) and 0.978 / 0.945 / 0.946 on the positive cells. The drop-one nested blends are:

| Model excluded | OOF log loss | Difference vs. full blend (0.01715) |
|---|---|---|
| ResNet | 0.01727 ± 0.00030 | +0.00012 |
| FT-Transformer | 0.01723 ± 0.00031 | +0.00008 |
| CatBoost | 0.01721 ± 0.00032 | +0.00006 |

Removing any one model worsens the nested score in all five folds (paired per-fold differences, from the scores printed in Step 14.5 and Step 15.3). The three models are strongly correlated and each adds 6–12 × 10⁻⁵ at most.

### Row-wise against compound-aware CV

The blend loss under the hybrid CV is higher than under the row-wise CV (0.01715 against 0.01607 on treated rows, +6.7%), consistent with the hybrid CV being less optimistic. This comparison is **not controlled**: run R used default hyperparameters and an earlier revision of the code, runs H1 and H2 used tuned hyperparameters, and the fold assignments differ. A controlled comparison (same code and hyperparameters under both protocols) was not run.

The diagnostics notebook, applied to the earlier hybrid run H0 and to the row-wise reference OOF, decomposes the paired per-row difference (+0.00114 on the blend) by compound frequency:

| Rows | Contribution to the difference | Loss, row-wise → hybrid |
|---|---|---|
| Frequent compounds (9.8% of the rows) | +0.00003 | 0.00172 → 0.00198 |
| All other rows | +0.00111 | 0.01762 → 0.01885 (+0.00123 per row, +7.0%; 95% interval over compounds [+0.00112, +0.00135]) |

Almost all of the difference is therefore on the rows of ordinary compounds, which the hybrid protocol evaluates as unseen. Under the hybrid CV the rows whose compound is in the training fold are the frequent compounds, with a loss of about 0.002 against about 0.0188 for the others (blend, run H2: 0.00198 and 0.01879; run H1: 0.00194 and 0.01878). These "seen" rows are not comparable to the 99.4% "seen" rows of the row-wise CV (blend loss 0.01604 seen, 0.01922 for the 0.6% unseen).

Other results of the diagnostics (run H0 unless stated):
- The worst fold (fold 4, +4.7% against the mean of the others) is not explained by the frequent compounds: the spread of the fold loss is 0.00091 on all rows and 0.00095 excluding them.
- The labels are identical across all rows of each of the 8 largest compounds, and 10 of the 87 targets with at least 50 positives take at least 50% of their positives from one compound (for example `proteasome_inhibitor`: 98.9% from `87d714366`, `hmgcr_inhibitor`: 86.9%, `nfkb_inhibitor`: 86.3%).
- Per-target error analysis (run H2 blend, Step 15.4): the 41 targets with 100 or more positives carry 47.9% of the loss (skill 0.342 against the per-target prior), the 53 targets with fewer than 20 positives 6.6% (skill +0.080, 95% interval [+0.056, +0.102]). The blend over-predicts the rare targets (expected over observed positives 1.47 below 20 positives, 7.7 below 5). Skill by number of positives, `row` against `hybrid`: 5–9 positives 0.064 → 0.123, 10–19 0.145 → 0.068, 20–49 0.121 → 0.061, 50–99 0.180 → 0.120, 100+ 0.374 → 0.342. The per-target baseline moves between protocols, so these skills are not strictly comparable. The Spearman correlation between positives per compound and the change in skill (+0.053, p = 0.47, 184 targets) is uninformative because the variable is almost constant (the median compound has 6 rows).

### Does tuning help under the compound-aware CV?

- **Run H2 against run H1. ** The blend changes by +1e-5 (0.01715 against 0.01714). Per model, the OOF changes by −9e-5 (ResNet), +8e-5 (FT-Transformer, which is the default configuration in C) and +1e-5 (CatBoost). The ResNet of H2 has a lower fold loss than that of H1 on all five folds when the three seeds are scored separately (mean difference −2.6e-4, from −1.7e-4 to −3.4e-4); after seed averaging the OOF gain shrinks to −9e-5, so most of the gain is variance that bagging already removes.
- **Offline recombination** (`moa-blend-offline`, run H2 against the earlier hybrid run H0): no variant (ResNet of H2 with FT-Transformer of H0, with or without the CatBoost of H0; averages of the two runs; six models with free weights) reaches the adoption threshold (gain of at least 5e-5 and improvement on at least 4 of 5 folds). The best variant, the six-model blend, scores 0.01714 against 0.01715 for H2 (H0 alone: 0.01720).
- The effect of tuning against the defaults, under the hybrid CV, was not measured in a controlled way.

### Do the test compounds overlap with the training compounds?

`moa-test-drug-overlap-nn` compares nearest-neighbour cosine distances (blocks `g-` and `c-`, standardised, pools of identical size) of the test rows against two references: validation rows whose compound is in the pool (row-wise folds) and rows whose compound is not (compound-grouped folds). The `cp_time` and `cp_dose` mixtures of train and test are the same (for example `cp_time` 24/48/72: 32.6/34.6/32.7% in train and 32.7/34.6/32.8% in test).

- **General overlap: not assessable.** The AUC between the two references is 0.516 (the distance hardly separates seen from unseen compounds), the mixture estimate of the share of "seen" rows is 0.506 with a bootstrap interval [−0.950, +2.072], and on pseudo-tests with a known share the estimate is off by up to 0.30. The estimate must be discarded. The separation exists only in the lowest quantiles (5th percentile of the distance: 0.121 seen, 0.164 unseen), that is, for near-duplicate rows of frequent compounds.
- **Frequent compounds.** For each compound with at least 20 rows a test row is *assigned* when its nearest neighbour belongs to it and the distance is below the 90th percentile of the distances of that compound's own rows. Assigned rows are compared with the number expected if the compound were absent (false-positive rate measured on the validation rows) and if it were present at its training frequency:

| Compound | Test rows assigned | Expected if absent | Expected if as in train | Estimated test share [95% CI] | Train share |
|---|---|---|---|---|---|
| `87d714366` | 133 | 5.3 | 111.5 | 3.93% [3.22, 4.72] | 3.27% |
| `8b87a7a83` | 23 | 0.5 | 30.5 | 0.69% [0.41, 0.98] | 0.92% |
| `d1b47f29d` | 24 | 0.5 | 26.9 | 0.72% [0.45, 1.03] | 0.81% |
| `d08af5d4b` | 36 | 11.4 | 37.9 | 0.83% [0.46, 1.27] | 0.89% |
| `292ab2c28` | 18 | 6.2 | 33.4 | 0.38% [0.12, 0.63] | 0.88% |
| `5628cb3ee` | 19 | 12.6 | 35.9 | 0.25% [−0.06, 0.62] | 0.92% |
| `9f80f3f77` | 46 | 40.3 | 64.8 | 0.26% [−0.36, 0.84] | 1.12% |
| `d50f18348` | 35 | 31.7 | 49.4 | 0.16% [−0.45, 0.76] | 0.85% |

The first five compounds are present at roughly their training frequency (together about 6.6% of the test rows against 6.8% of the training rows); the last three share their mechanism with other compounds, so the rule cannot separate them (false-positive rate 0.9–1.1%). The ninth frequent compound (19 rows) was not analysed. This is an indirect estimate and does not prove the presence or absence of a specific compound.

### Nearest-neighbour override for frequent compounds (negative result)

Since the frequent compounds appear in the test set, `moa-knn-override-freq-drugs` tested on the run H2 OOF (hybrid folds, parameters estimated on the training part of each fold only) whether pulling the predictions of rows assigned to a frequent compound towards its label profile reduces the loss. The ceiling on the gain is the current loss on those rows, 1.68e-4 on the mean over all rows. The assignment rule has a validation precision of 0.722 (663 of 2,381 assigned rows belong to another compound) and a recall of 0.809. All four variants worsen the OOF loss, on all five folds:

| Variant | Change in OOF loss (positive = worse) | On correctly assigned rows | On false positives |
|---|---|---|---|
| replace with the profile | +1.47e-3 | +0.84e-4 | +13.9e-4 |
| mixture weighted by the estimated precision | +2.6e-4 | +0.70e-4 | +1.9e-4 |
| mixture 0.5 | +2.1e-4 | +0.40e-4 | +1.7e-4 |
| mixture 0.9 | +6.4e-4 | +0.75e-4 | +5.7e-4 |

The compounds with high precision (0.95 or more) account for about 3.5e-5 of the ceiling; the rest sits in compounds with a precision of about 0.4 (`d50f18348`, `9f80f3f77`). The override was not adopted.

### Independent check on the Kaggle test set

Two models were submitted as late submissions through the inference notebook: the row-wise models of run R and the hybrid models of run H2. The Kaggle test set was never used to tune hyperparameters or to choose among configurations. It did, however, motivate the introduction of the compound-aware protocols (the first submission showed a gap with the CV estimate), and its unlabeled features were used in the nearest-neighbour analysis above.

| Estimate | Run R | Run H2 |
|---|---|---|
| Nested-CV blend, treated rows only | 0.01607 ± 0.00017 | 0.01715 ± 0.00031 |
| Nested-CV blend, rescaled to all test rows (× 0.910) | 0.01462 | 0.01561 |
| Kaggle private score | **0.01634** | **0.01643** |
| Kaggle public score | 0.01865 | 0.01873 |
| Private score above the rescaled CV estimate | +11.8% | +5.3% |
| Public score above the rescaled CV estimate | +27.5% | +20.0% |

The cross-validation scores are computed on treated rows only, whereas the Kaggle score averages over all test rows. Control rows are predicted as 0 with true value 0 and add almost no loss, so the all-rows score is approximately the treated-only score multiplied by the share of treated rows. That share is 0.910 in the public test set, which gives, for example, 0.01607 × 0.910 = 0.01462. The share in the hidden part of the test set may differ slightly, so the rescaling is approximate. The private score, computed on more rows, is the more reliable of the two. The consistency of the inference notebook was checked on the public test set: its predictions differ from those of the training notebook by at most 1.1e-16 (run R).

How to read these numbers:
- The hybrid CV reproduces about 57% of the gap between the row-wise CV estimate and the private score (training notebook, Step 15.5). The hybrid estimate of the models of H2 (0.01561) is still 5.3% below the private score.
- The two submissions differ by +9e-5 (private) and +8e-5 (public) in favour of the row-wise models. With one submission per configuration and a fold standard deviation of 3e-4, this difference cannot be interpreted as one model being worse: the conclusion that holds is that changing the protocol, the tuning and the model selection did not move the test score measurably.
- *Back-of-the-envelope inference, not a measurement.* If about 6.6% to 9.8% of the test rows belong to frequent compounds (the five compounds found above, up to all nine at their training frequency) and the losses of the hybrid CV (about 0.002 on those rows, 0.0188 on the others) held on the test set, the all-rows estimate would be 0.0156–0.0161, so the residual gap with the private score (0.01643) would be 2–5%. The public score (0.01873) stays about 20% above the CV estimate, and the public and private scores differ by 0.0023 for both submissions: the test subsets differ in difficulty, and this cannot be explained by compound overlap.

### Ablation: is an extra network worth it?

Run on the row-wise protocol with default hyperparameters (run R). Three configurations with a fourth model were also run on the same folds. The base is always ResNet + FT-Transformer + CatBoost. The ablation was run on an earlier revision of the code, before the final fixes (among them the smoothed prior for constant CatBoost targets), so its absolute values differ from the final run in the last digit.

| Configuration | Extra net OOF | Ensemble (nested) | Extra net weight | Total run |
|---|---|---|---|---|
| + MLP (PCA features) | 0.01649 | 0.01604 | 0.015 | 113 min |
| + 1D-CNN (PCA features) | 0.01661 | 0.01605 | 0.082 | 113 min |
| + 1D-CNN (raw features) | 0.01661 | 0.01606 | 0.070 | 106 min |
| No extra network (baseline at the time of the ablation) | – | 0.01606 | – | 129 min |
| **No extra network (final rerun)** | – | **0.01605** | – | 131 min |

The MLP was redundant with the ResNet (same features, same model family) and was replaced by a 1D-CNN with a different inductive bias. The CNN earns 7–8% of the weight, but no configuration improves the ensemble: all scores lie within 0.00002, below the fold standard deviation (0.00017) and the run-to-run GPU noise (about 1e-5). Giving the CNN raw instead of PCA features did not change its own OOF score (0.01661). The simpler 3-model blend was therefore kept as the final solution: fewer components to train, explain and maintain, for the same score.

The training notebooks containing the MLP and 1D-CNN are in [`software/ablated-models/`](merlonghi-moa-prediction/software/ablated-models/). The figures above are reported from those runs and can be reproduced from the repository alone.

The "differences below the fold standard deviation" argument is unpaired. The per-fold scores of the ablation runs were not kept, so a paired comparison (as done above for CatBoost) was not possible; the conclusion therefore rests on the differences being of the same order as the run-to-run noise.

### Protocol note

An earlier version of the code selected the best epoch on the same fold that produced the OOF predictions, giving an ensemble score of 0.01595. The final code uses an inner validation split instead, and the score moved to about 0.0161. The difference of about 0.0001 is consistent with selection optimism, but it also includes the effect of training on 10% less data per fold and the run-to-run noise (about 1e-5), so it is not an exact measurement of the optimism. The first tuning notebook (v1) has the same flaw by construction: the best epoch of each trial is selected on the validation fold that is scored, so its fold-mean losses (ResNet 0.01602, FT-Transformer 0.01698, CatBoost 0.01767) are not comparable with the OOF scores. Versions v2 and v3 use an inner split.

### Leaderboard

The row-wise submission (private score 0.01634) would rank approximately 1400th of 4,373 in the private leaderboard and 1800th in the public one. Late submissions do not appear on the leaderboards, so this is an approximation obtained by comparing the scores with the final rankings of the competition. The hybrid submission scores 9e-5 higher on the private set; its rank was not computed.

## Limitations

- **Compound grouping in the CV is a design choice, and the test overlap is partly unknown.** The training compounds are known (`train_drug.csv`) but the test set has no identifier. `HYBRID_MIN_ROWS` was chosen on the fold-size balance of a single splitter seed; the thresholds 30, 60 and 100 produce identical splits, and the dependence of the CV loss on the threshold was not measured. The nearest-neighbour analysis shows that frequent compounds are in the test set but cannot say anything about ordinary compounds (AUC 0.516); neither the hybrid nor the pure grouped CV can be declared the faithful one, and their difference brackets the uncertainty.
- **The row-wise and hybrid results are not directly comparable.** Run R used default hyperparameters and an earlier revision of the code; H1 and H2 used tuned hyperparameters. The +6.7% difference mixes the effect of the protocol with these changes.
- **Selection bias from tuning and from model choice.** Hyperparameters, architectures and the final 3-model choice were selected by looking at OOF scores on folds that overlap with the reported ones: the tuning folds (folds 0 and 1 of the tuning seeds) contain rows that are also in the validation folds of the reported OOF, and the recombinations of the OOF of two runs were all scored on the same OOF. No hold-out was reserved before tuning, and nested cross-validation of the whole procedure was too costly (estimated at tens of hours on the available GPUs). The CV scores are therefore optimistic by an unknown amount, and the differences between the configurations that were compared are below the fold standard deviation.
- **Truncated and noisy tuning.** The search stages use one seed per trial and 25/15/15 trials (ResNet/CatBoost/FT-Transformer), of which 10/4/4 are random start-up trials of the TPE sampler. In v3 several tuned values lie at or near the bounds of the search space (ResNet dropout, batch size and learning rate; CatBoost `l2_leaf_reg`), so the optimum may be outside it. Tuning v2 used row-wise folds, which does not match the hybrid evaluation protocol. The adoption rules protect against adopting noise (confirmation on a second fold; in v3, three paired seeds) but do not prove that the adopted values are better than the defaults in the hybrid CV.
- **Preprocessing fitted on the whole training fold.** The RankGauss and PCA transforms are fitted on the entire training fold, including the 10% inner validation rows. The transforms are unsupervised, so no labels leak and the outer fold is never seen; the only possible effect is a slightly optimistic early-stopping choice.
- **CV on treated rows only.** CV scores are comparable with leaderboard scores only after rescaling by the treated share (see above).
- **Two submissions, noisy test subsets.** Each Kaggle score comes from a single submission. Public and private scores differ by 0.0023 for both submissions, more than seven times the fold standard deviation of the hybrid CV (0.00031), so the fold standard deviation understates the uncertainty on unseen data. Late submissions do not appear on the leaderboard. The two submissions cannot be ranked against each other (difference 9e-5).
- **Frequent-compound analyses are descriptive.** The test-compound analysis and the back-of-the-envelope reweighting assume that the CV losses on seen and unseen rows hold on the test set, and rely on distances that identify only near-duplicate rows.
- **Non-determinism.** Repeated GPU runs of the row-wise code differed by about 1e-5 in OOF log loss; the run-to-run noise of the hybrid runs was not measured.
- **No ablation** of individual design choices (PCA vs. raw features for the ResNet, label smoothing) was run. The only raw-vs-PCA comparison concerns the 1D-CNN of the extra-network ablation.
- **Mixed precision in validation, full precision at test time.** Out-of-fold predictions, and therefore the blend weights, are computed with fp16 autocast on GPU, while the inference notebook runs in fp32. On fold 0 (first seed) of run H2 the two modes differ by at most 5.72e-4 (ResNet) and 4.30e-3 (FT-Transformer) in predicted probability, but the log loss differs by at most 1e-6 (0.017124 vs. 0.017124 and 0.017690 vs. 0.017689, Step 15.1). The effect on the blend weights was not measured separately.
- **Inference depends on saved artifacts.** The inference notebook needs the Kaggle Dataset of trained models, which is not stored in this repository; it is produced by the training notebook (see [`software/README.md`](merlonghi-moa-prediction/software/README.md)). Exactly one attached input may contain `metadata.json`, so analysis Datasets with a copy of it must not be attached to the inference notebook.

## Repository structure

```
merlonghi-moa-prediction/
├── README.md                             <- this file
├── software/
│     ├── README.md                       <- setup / how-to-run instructions
│     ├── moa-training.ipynb              <- training and cross-validation (CV_SCHEME: hybrid / grouped / row)
│     ├── moa-training.pdf                <- landscape export of the training notebook
│     ├── moa-inference.ipynb             <- inference and Kaggle submission
│     ├── moa-inference.pdf               <- landscape export of the inference notebook
│     ├── moa-tuning.ipynb                <- tuning v1 (row-wise folds, single stage)
│     ├── moa-tuning-row.ipynb            <- tuning v2 (row-wise folds, two stages)
│     ├── moa-tuning-hybrid.ipynb         <- tuning v3 (compound-aware folds, paired seeds)
│     ├── moa-fold-diagnostics.ipynb      <- fold and compound diagnostics of a saved run
│     ├── moa-blend-offline.ipynb         <- offline recombination of the OOF of two runs
│     ├── moa-test-drug-overlap-nn.ipynb  <- indirect test-compound-overlap analysis
│     ├── moa-knn-override-freq-drugs.ipynb <- offline test of the nearest-neighbour override
│     ├── requirements.txt                <- python packages to install
│     ├── requirements-frozen.txt         <- versions of the reported training run
│     └── ablated-models/
│           ├── moa-training-with-MoAMLP.ipynb                 <- training notebook version with ablated MLP
│           └── moa-training-with-MoAConv1D.ipynb              <- training notebook version with ablated 1D-CNN
└── doc/
      ├── moa-presentation.pdf
      └── moa-presentation.tex            <- latex beamer
```

## References

1. Kaggle, *Mechanisms of Action (MoA) Prediction*, 2020. <https://www.kaggle.com/competitions/lish-moa>
2. Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data.* NeurIPS 2021. (FT-Transformer)
3. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS 2018.
4. Sechidis, K., Tsoumakas, G., Vlahavas, I. (2011). *On the Stratification of Multi-Label Data.* ECML PKDD 2011.
5. Bradberry, T. *iterative-stratification* (Python package implementing [4]). <https://github.com/trent-b/iterative-stratification>
6. Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019. (AdamW)
7. Micikevicius, P. et al. (2018). *Mixed Precision Training.* ICLR 2018.
8. Szegedy, C. et al. (2016). *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. (label smoothing)
9. Nelder, J.A., Mead, R. (1965). *A simplex method for function minimization.* The Computer Journal 7(4).
10. Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
11. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12. (`QuantileTransformer`, `PCA`)
12. Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD 2019.
13. Bergstra, J., Bardenet, R., Bengio, Y., Kégl, B. (2011). *Algorithms for Hyper-Parameter Optimization.* NeurIPS 2011. (Tree-structured Parzen Estimator)