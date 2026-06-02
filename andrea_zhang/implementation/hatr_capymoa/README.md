# HATR — CapyMOA wrapper

CapyMOA wrapper for `moa.classifiers.trees.HoeffdingAdaptiveTreeRegressor`

## Files

- `_hatr.py` — the `HoeffdingAdaptiveTreeRegressor` wrapper class (subclass of
  `capymoa.base.MOARegressor`), written in the style of the existing wrappers
  (`capymoa/regressor/_fimtdd.py`, `_arffimtdd.py`).
- `__init__.py` — exports the class.

For build, usage, and CapyMOA contribution instructions see [`../README.md`](../README.md).

## River equivalence notes

- `leaf_prediction="mean"`, `bootstrap_sampling=False`: equivalent to River.
- With `bootstrap_sampling=True` the RNGs differ (Python vs Java).
- The wrapper defaults same as in River (`leaf_prediction="adaptive"`,
  `bootstrap_sampling=True`, `merit_preprune=True`).

## Parameter → MOA CLI flag mapping

| Python parameter | MOA flag |
|---|---|
| grace_period | -g |
| split_confidence (delta) | -d |
| tie_threshold (tau) | -t |
| leaf_prediction | -l (MEAN/MODEL/ADAPTIVE) |
| model_selector_decay | -e |
| bootstrap_sampling | -b (flag) |
| drift_window_threshold | -w |
| switch_significance | -s |
| min_samples_split | -m |
| max_depth | -D (-1 = unlimited) |
| tebst_digits | -z |
| merit_preprune | -p (flag) |
| adwin_delta | -A |
| learning_ratio | -L |
| max_size_mb | -M |
| memory_estimate_period | -E |
| stop_mem_management | -S (flag) |
| remove_poor_attrs | -R (flag) |
| random_seed | via `setRandomSeed` (MOARegressor) |
