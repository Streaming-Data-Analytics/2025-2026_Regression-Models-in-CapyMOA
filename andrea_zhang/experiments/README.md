# Experiments

Cross-framework comparison of HATR between **River** and **CapyMOA** (Parts 2 and 3 of the project).

## Notebooks

| Notebook | Description |
|---|---|
| `hatr-comparison.ipynb` | Qualitative comparison — River vs CapyMOA on Fried and Bike datasets |
| `hatr-crossframework-quality.ipynb` | Regression quality metrics (MAE, RMSE) across leaf modes and datasets |
| `hatr-crossframework-performance.ipynb` | Computational performance — training time and memory usage |
| `hatr-full-datasets.ipynb` | River vs CapyMOA on full datasets (no instance cap) with default configuration (`leaf_prediction="adaptive"`, bootstrap) |

## Datasets

The notebooks use:

| Dataset | Source | Instances | Features |
|---|---|---|---|
| Fried (Friedman) | `capymoa.datasets.Fried` | ~40 768 | 10 |
| Bike Sharing | `capymoa.datasets.Bike` | ~17 379 | 12 |
| HyperPlaneRegression | `capymoa.stream.generator.HyperPlaneRegression` | 50 000 | 10 |
