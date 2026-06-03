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

The crossframework notebooks cap all datasets at `MAX_INSTANCES = 17 000` for uniform comparison.
`hatr-full-datasets.ipynb` uses the natural sizes listed below.

| Dataset | Source | Instances | Features | Drift |
|---|---|---|---|---|
| Fried (Friedman) | `capymoa.datasets.Fried` | ~40 768 | 10 | none |
| Bike Sharing | `capymoa.datasets.Bike` | ~17 379 | 12 | natural (temporal) |
| Synthetic | `utils.make_synthetic` | configurable | 5 | abrupt at midpoint |
| HyperPlane (slow) | `capymoa.stream.generator.HyperPlaneRegression` | 50 000 | 10 | gradual (magnitude=0.001, 2 attrs) |
| HyperFast | `capymoa.stream.generator.HyperPlaneRegression` | 50 000 | 10 | rapid (magnitude=0.1, 5 attrs) |
| FriedDrift | `utils.make_fried_drift` | 40 000 | 10 | abrupt at midpoint (Friedman relation changes from x0–x4 to x5–x9) |
| RBFReg | `utils.make_rbf_regression` | 30 000 | 10 | gradual after midpoint (RBF centroids shift) |
| ElectricityDemand | `utils.load_electricity_demand` (`data/electricity.arff`) | ~45 312 | 7 | natural (temporal, NSW demand) |
