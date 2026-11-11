# CV-scheme runs — the three protocols with the same code and hyperparameters

This folder holds two copies of [`../moa-training.ipynb`](../moa-training.ipynb), each run with the same code, the same seeds and the same tuned hyperparameters ([`../hyperparameters/best_hyperparameters_hybrid.json`](../hyperparameters/best_hyperparameters_hybrid.json)) as run H2 (the `hybrid` run saved in the delivered training notebook). Only `CV_SCHEME` differs, so R3, H2 and G3 are the controlled comparison of the three cross-validation protocols (the runs are defined in the [main README](../../README.md#runs-referred-to-in-the-results)).

| Run | Notebook | `CV_SCHEME` | Validation rows per fold | Nested blend (treated rows) | Total run time (2× T4) |
|---|---|---|---|---|---|
| R3 | [`moa-training-row.ipynb`](moa-training-row.ipynb) ([PDF](moa-training-row.pdf)) | `"row"` | 4,390 / 4,390 / 4,389 / 4,390 / 4,389 | 0.01595 ± 0.00016 | 116.5 min |
| H2 | [`../moa-training.ipynb`](../moa-training.ipynb) ([PDF](../moa-training.pdf)) | `"hybrid"` | 4,386 / 4,356 / 4,371 / 4,383 / 4,452 | 0.01715 ± 0.00031 | 95.0 min |
| G3 | [`moa-training-grouped.ipynb`](moa-training-grouped.ipynb) ([PDF](moa-training-grouped.pdf)) | `"grouped"` | 3,987 / 4,136 / 5,512 / 3,933 / 4,380 | 0.01926 ± 0.00099 | 89.9 min |

## Notes

- **How to run.** Like the training notebook (section 5 of [`../README.md`](../README.md)): import the notebook on Kaggle, attach the competition data and the file `best_hyperparameters_hybrid.json` as inputs, set `HYPERPARAMETERS_FILE`, select 2× T4 and the Internet toggle on, and run all cells.
- **Step 15.5 compares with run R, not with R3.** In H2 and G3 the reference OOF folder (`REFERENCE_OOF_DIR`) holds the OOF predictions of run R (row-wise, default hyperparameters, earlier code revision), so the comparison printed there mixes protocol, hyperparameters and code revision. In R3 `REFERENCE_OOF_DIR` is `None`: Step 15.5 prints only the loss on rows whose compound is, or is not, in the training fold and states that the gap to the Kaggle score is not reproduced by construction. The controlled comparison is made with [`../moa-fold-diagnostics.ipynb`](../moa-fold-diagnostics.ipynb), section 7 (`RUN_DIR` = H2, `ROW_RUN_DIR` = R3, `EXTRA_RUN_DIRS` = G3).
- **Grouped folds are unbalanced.** The G3 notebook prints a warning in Step 14.2b (a fold deviates by more than 15% from the mean size); 34–36 of the 206 targets have no positive in a validation fold. The nested blend is weighted by the validation rows of each fold.
- **Reading the grouped result.** The G3 estimate (0.01926) is above the Kaggle private scores of the submitted models (about 0.0180 on the treated-rows scale), so it is a pessimistic bound rather than a faithful estimate.
- **Hyperparameters may favour `hybrid`.** The tuned values were selected on compound-aware (`hybrid`) folds, so R3 and G3 may be slightly disadvantaged against H2.
