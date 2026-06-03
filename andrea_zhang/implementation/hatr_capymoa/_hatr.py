from typing import Optional

from capymoa.base import MOARegressor
from capymoa.stream._stream import Schema
from moa.classifiers.trees import (
    HoeffdingAdaptiveTreeRegressor as _MOA_HoeffdingAdaptiveTreeRegressor,
)

_LEAF_PREDICTION = {"mean": "MEAN", "model": "MODEL", "adaptive": "ADAPTIVE"}


class HoeffdingAdaptiveTreeRegressor(MOARegressor):
    """Hoeffding Adaptive Tree Regressor (HATR).

    Regression Hoeffding Tree with ADWIN-based concept drift detection.
    Each internal node runs an ADWIN detector on prediction error; on drift
    an alternate subtree grows in the background and a z-test decides when
    to swap it in place of the current subtree.

    Numerically equivalent to River's ``HoeffdingAdaptiveTreeRegressor`` in
    the deterministic configuration (``leaf_prediction="mean"``,
    ``bootstrap_sampling=False``).  With bootstrap sampling only aggregate
    metrics are comparable because the RNGs differ.

    Reference: Bifet & Gavaldà, IDA 2009.
    https://doi.org/10.1007/978-3-642-03915-7_22

    Parameters
    ----------
    schema :
        Stream schema.
    grace_period :
        Instances a leaf observes between split attempts.
    split_confidence :
        Significance level (delta) for the Hoeffding bound.
    tie_threshold :
        Force a split to break ties when the bound falls below this value.
    leaf_prediction :
        ``"mean"``, ``"model"`` (perceptron), or ``"adaptive"``.
    model_selector_decay :
        EWMA decay for the adaptive leaf's model selector.
    bootstrap_sampling :
        Poisson(1) bootstrap sampling in leaf updates.
    drift_window_threshold :
        Minimum instances the alternate tree must see before the z-test runs.
    switch_significance :
        p-value threshold for the z-test swap decision.
    min_samples_split :
        Minimum samples per branch for a split candidate to be valid.
    max_depth :
        Maximum tree depth; ``None`` means unlimited.
    tebst_digits :
        Rounding digits for the Truncated E-BST numeric observer.
    merit_preprune :
        Enable merit-based pre-pruning.
    adwin_delta :
        Delta parameter for per-node ADWIN detectors.
    learning_ratio :
        Learning rate for the perceptron leaf model.
    max_size_mb :
        Memory limit in MB (requires SizeOf agent; silently disabled otherwise).
    memory_estimate_period :
        Instances between memory checks.
    stop_mem_management :
        Stop growing (instead of deactivating leaves) when the limit is hit.
    remove_poor_attrs :
        Disable attributes with consistently poor merit.
    random_seed :
        Seed for bootstrap sampling.

    Example usage (requires the HATR JAR on the MOA classpath)::

        from capymoa.datasets import Fried
        from capymoa.regressor import HoeffdingAdaptiveTreeRegressor
        from capymoa.evaluation import prequential_evaluation

        stream = Fried()
        learner = HoeffdingAdaptiveTreeRegressor(stream.get_schema())
        results = prequential_evaluation(stream, learner, max_instances=1000)
        print(results["cumulative"].rmse())
    """

    def __init__(
        self,
        schema: Schema,
        grace_period: int = 200,
        split_confidence: float = 1.0e-7,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "adaptive",
        model_selector_decay: float = 0.95,
        bootstrap_sampling: bool = True,
        drift_window_threshold: int = 300,
        switch_significance: float = 0.05,
        min_samples_split: int = 5,
        max_depth: Optional[int] = None,
        tebst_digits: int = 1,
        merit_preprune: bool = True,
        adwin_delta: float = 0.002,
        learning_ratio: float = 0.01,
        max_size_mb: float = 500.0,
        memory_estimate_period: int = 1_000_000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        leaf = leaf_prediction.lower()
        if leaf not in _LEAF_PREDICTION:
            raise ValueError(
                f"Invalid leaf_prediction '{leaf_prediction}'. "
                f"Expected one of {list(_LEAF_PREDICTION)}."
            )

        cli = []
        cli.append(f"-g {grace_period}")
        cli.append(f"-d {split_confidence}")
        cli.append(f"-t {tie_threshold}")
        cli.append(f"-l {_LEAF_PREDICTION[leaf]}")
        cli.append(f"-e {model_selector_decay}")
        if bootstrap_sampling:
            cli.append("-b")
        cli.append(f"-w {drift_window_threshold}")
        cli.append(f"-s {switch_significance}")
        cli.append(f"-m {min_samples_split}")
        cli.append(f"-D {max_depth if max_depth is not None else -1}")
        cli.append(f"-z {tebst_digits}")
        if merit_preprune:
            cli.append("-p")
        cli.append(f"-A {adwin_delta}")
        cli.append(f"-L {learning_ratio}")
        cli.append(f"-M {max_size_mb}")
        cli.append(f"-E {memory_estimate_period}")
        if stop_mem_management:
            cli.append("-S")
        if remove_poor_attrs:
            cli.append("-R")

        self.moa_learner = _MOA_HoeffdingAdaptiveTreeRegressor()

        super().__init__(
            schema=schema,
            CLI=" ".join(cli),
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )
