# Hyperparameters — MoA Prediction

This folder contains the hyperparameters read by the [training notebook](../moa-training.ipynb) (Step 1.5, constants `USE_TUNED_HYPERPARAMETERS` and `HYPERPARAMETERS_FILE`). The following table shows which file belongs to which run (the runs are defined in the [main README](../../README.md#runs-referred-to-in-the-results)).

| Configuration | Runs | Produced by | JSON file |
|---|---|---|---|
| Defaults | `R`, `H0`, `G0` | – | [`default_hyperparameters.json`](default_hyperparameters.json) |
| Tuned | `H2`, `R3`, `G3` | [`moa-tuning-hybrid.ipynb`](../moa-tuning-hybrid.ipynb) | [`best_hyperparameters_hybrid.json`](best_hyperparameters_hybrid.json) |

`R3`, `H2` and `G3` use the same JSON file and the same code and differ only in `CV_SCHEME` (`row`, `hybrid`, `grouped`), so their scores are comparable with one another.

## Values

| Model | Parameter | Default | Tuned |
|---|---|---|---|
| ResNet | learning rate | 1e-3 | 2.67e-3 |
| ResNet | dropout | 0.2 | 0.583 |
| ResNet | weight decay | 1e-5 | 2.66e-4 |
| ResNet | hidden units / batch size | 512 / 256 | 768 / 512 |
| FT-Transformer | all parameters | default | default (see below) |
| CatBoost | depth / `l2_leaf_reg` | 5 / 3 | 4 / 56.0 |
| CatBoost | learning rate / subsample | 0.08 / none | 0.1034 / 0.829 |
| CatBoost | iterations | 250 | 250 |

- **FT-Transformer:** no candidate beat the default on the confirmation fold of the tuning notebook, so the tuned file contains the default values for this model.
- **Bounds:** the tuned ResNet dropout (0.583, range [0.1, 0.6]), batch size (512, the largest value) and learning rate (2.67e-3, range [3e-4, 3e-3]) lie at or near the upper bounds of the search space, and the CatBoost `l2_leaf_reg` sits near the upper end of its range ([1, 60]), so the optimum may be outside it.
- **Selection:** the values were selected on folds 0 and 1 of a compound-aware split (seed 1234) that overlap the validation folds of the reported CV, so the CV scores of the tuned runs are optimistic by an unknown amount. A first exploratory tuning cycle on row-wise folds was run before the final notebook; that notebook is not preserved in this repository and none of its results is reported, but it informed the search space.
- **Other protocols:** the tuned values were chosen on `hybrid` folds, so they may favour the `hybrid` protocol more than `row` or `grouped`.
