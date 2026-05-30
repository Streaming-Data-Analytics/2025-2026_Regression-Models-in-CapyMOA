from typing import Optional

from capymoa.base import MOARegressor
from capymoa.stream._stream import Schema
from moa.classifiers.trees import (
    HoeffdingAdaptiveTreeRegressor as _MOA_HoeffdingAdaptiveTreeRegressor,
)

_LEAF_PREDICTION = {"mean": "MEAN", "model": "MODEL", "adaptive": "ADAPTIVE"}


class HoeffdingAdaptiveTreeRegressor(MOARegressor):
    """Hoeffding Adaptive Tree Regressor (HATR).

    A regression Hoeffding Tree that uses an ADWIN concept-drift detector at each
    decision node to monitor changes in the data distribution. When a drift is
    detected in a node, an alternate subtree is grown in the background; once it has
    seen enough instances, a z-test decides whether to replace the current subtree
    with the (significantly better) alternate one.

    This is a native MOA port of River's ``HoeffdingAdaptiveTreeRegressor`` and is
    numerically equivalent to it in the deterministic configuration
    (``leaf_prediction="mean"``, ``bootstrap_sampling=False``). With bootstrap
    sampling the two libraries cannot match instance-by-instance because the random
    number generators differ (Python's Mersenne Twister vs Java's
    ``java.util.Random``); only aggregate metrics are comparable.

    Reference:

    `Bifet, Albert, and Ricard Gavaldà. "Adaptive learning from evolving data
    streams." International Symposium on Intelligent Data Analysis, 2009.
    <https://doi.org/10.1007/978-3-642-03915-7_22>`_

    Example usage (requires the HATR class on the MOA classpath)::

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
        random_seed: Optional[int] = None,
    ) -> None:
        """Construct a Hoeffding Adaptive Tree Regressor.

        :param schema: The schema of the stream.
        :param grace_period: Number of instances a leaf should observe between split attempts.
        :param split_confidence: Significance level (delta) for the Hoeffding bound; values
            close to 0 imply longer split-decision delays.
        :param tie_threshold: Threshold below which a split is forced to break ties.
        :param leaf_prediction: Prediction mechanism at the leaves: ``"mean"`` (target mean),
            ``"model"`` (online linear model), or ``"adaptive"`` (chooses dynamically).
        :param model_selector_decay: Decay factor of the faded squared errors used by the
            adaptive leaf to choose between mean and model predictions.
        :param bootstrap_sampling: If True, apply Poisson(1) bootstrap sampling in the leaves.
        :param drift_window_threshold: Minimum number of instances an alternate tree must
            observe before being considered as a replacement.
        :param switch_significance: p-value threshold of the z-test used to swap a subtree
            with its alternate tree.
        :param min_samples_split: Minimum number of samples each branch of a split candidate
            must have for the split to be valid.
        :param max_depth: Maximum tree depth. ``None`` means unlimited.
        :param tebst_digits: Number of decimal places used to round feature values in the
            Truncated E-BST numeric attribute observer.
        :param merit_preprune: If True, enable merit-based pre-pruning (null-split option).
        :param adwin_delta: Delta parameter of the per-node ADWIN drift detectors.
        :param learning_ratio: Learning rate of the linear (perceptron) leaf model.
        :param max_size_mb: Maximum memory consumed by the tree in MB. Requires the
            SizeOf agent (``-javaagent:sizeofag.jar`` and ``--add-opens`` flags); silently
            disabled otherwise.
        :param memory_estimate_period: Number of instances between memory consumption
            checks. Only relevant when ``max_size_mb`` is active.
        :param stop_mem_management: If True, stop growing the tree (rather than
            deactivating leaves) when the memory limit is hit.
        :param random_seed: Random seed for reproducibility (used by bootstrap sampling).
        """
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
        cli.append("-b") if bootstrap_sampling else None
        cli.append(f"-w {drift_window_threshold}")
        cli.append(f"-s {switch_significance}")
        cli.append(f"-m {min_samples_split}")
        cli.append(f"-D {max_depth if max_depth is not None else -1}")
        cli.append(f"-z {tebst_digits}")
        cli.append("-p") if merit_preprune else None
        cli.append(f"-A {adwin_delta}")
        cli.append(f"-L {learning_ratio}")
        cli.append(f"-M {max_size_mb}")
        cli.append(f"-E {memory_estimate_period}")
        cli.append("-S") if stop_mem_management else None

        self.moa_learner = _MOA_HoeffdingAdaptiveTreeRegressor()

        super().__init__(
            schema=schema,
            CLI=" ".join(cli),
            random_seed=random_seed,
            moa_learner=self.moa_learner,
        )
