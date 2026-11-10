# Hyperparameters — MoA Prediction
This folder contains the hyperparameters used in the [training notebook](../moa-training.ipynb). The following table shows which tuning version corresponds to each file.


| Tuning Version | Run | Tuning Notebook | JSON file |
|---|---|---|---|
| Default | `R` and `H0` | - | [`default_hyperparameters.json`](default_hyperparameters.json) |
| Tuning v2 | `H1` | [`moa-tuning-row.ipynb`](../moa-tuning-row.ipynb) | [`default_hyperparameters_row.json`](default_hyperparameters_row.json) |
| Tuning v3 | `H2` | [`moa-tuning-hybrid.ipynb`](../moa-tuning-hybrid.ipynb) | [`default_hyperparameters_hybrid.json`](default_hyperparameters_hybrid.json) |

Values of H1 come from the original execution; a rerun of tuning v2 gives slightly different values (GPU non-determinism).