from typing import List, Literal, Optional

from capymoa.base import MOARegressor
from capymoa.stream._stream import Schema

from moa.classifiers.trees import (
    HoeffdingAdaptiveTreeRegressor as _MOA_HATR,
)

_LEAF_PREDICTION = Literal["mean", "model", "adaptive"]

_LEAF_PREDICTION_MAP = {
    "mean": "MEAN",
    "model": "MODEL",
    "adaptive": "ADAPTIVE",
}


class HoeffdingAdaptiveTreeRegressor(MOARegressor):
    def __init__(
        self,
        schema: Schema,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: _LEAF_PREDICTION = "adaptive",
        learning_rate: float = 0.01,
        l2: float = 0.0,
        l1: float = 0.0,
        model_selector_decay: float = 0.95,
        min_samples_split: int = 5,
        drift_window_threshold: int = 300,
        switch_significance: float = 0.05,
        adwin_delta: float = 0.002,
        max_size: float = 500.0,
        memory_estimate_period: int = 1_000_000,
        tebst_digits: int = 1,
        numerical_multiway: bool = False,
        numerical_multiway_radius: float = 0.25,
        max_depth: Optional[int] = None,
        no_bootstrap_sampling: bool = False,
        binary_split: bool = False,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        no_pre_prune: bool = False,
        nominal_attributes: Optional[List[int]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        :param grace_period: Instances a leaf observes between split attempts.
        :param delta: Hoeffding bound confidence (lower = more confident splits).
        :param tau: Tie-breaking threshold.
        :param leaf_prediction: "mean", "model", or "adaptive".
        :param learning_rate: SGD learning rate for leaf linear models.
        :param l2: L2 regularisation for leaf models (mutually exclusive with l1).
        :param l1: L1 regularisation for leaf models (mutually exclusive with l2).
        :param model_selector_decay: FMSE decay for adaptive leaf selection.
        :param min_samples_split: Min samples per child branch after a split.
        :param drift_window_threshold: Min examples an alternate tree must see
            before being eligible for replacement.
        :param switch_significance: p-value threshold for the z-test when
            promoting an alternate tree.
        :param adwin_delta: ADWIN delta for all drift detectors.
        :param max_size: Maximum tree size in MiB.
        :param memory_estimate_period: Instances between memory size checks.
        :param tebst_digits: Decimal digits for TEBST rounding (default numerical splitter).
        :param numerical_multiway: Enable multiway splits on numerical features via RadiusSplitter.
            Replaces TEBSTSplitter; scale features first (River QOSplitter default: radius=0.25).
        :param numerical_multiway_radius: Bin width for RadiusSplitter: slot = floor(x / radius).
        :param max_depth: Maximum tree depth (None = unlimited).
        :param no_bootstrap_sampling: Disable Poisson bootstrap sampling.
        :param binary_split: Force binary splits for nominal features.
        :param stop_mem_management: Stop growing when memory limit is hit.
        :param remove_poor_attrs: Disable attributes with poor split merit.
        :param no_pre_prune: Disable merit-based pre-pruning.
        :param nominal_attributes: List of 0-based attribute indices to treat as
            nominal (mirrors River's nominal_attributes). None means use schema type only.
        :param random_seed: RNG seed (None = no fixed seed).
        """
        mapping = _LEAF_PREDICTION_MAP.get(leaf_prediction.lower())
        if mapping is None:
            raise ValueError(
                f"Invalid leaf_prediction '{leaf_prediction}'. "
                f"Choose one of: {list(_LEAF_PREDICTION_MAP.keys())}."
            )

        cli = []
        cli.append(f"-g {grace_period}")
        cli.append(f"-c {delta}")
        cli.append(f"-t {tau}")
        cli.append(f"-l {mapping}")
        cli.append(f"-r {learning_rate}")
        cli.append(f"-2 {l2}")
        cli.append(f"-1 {l1}")
        cli.append(f"-q {model_selector_decay}")
        cli.append(f"-m {min_samples_split}")
        cli.append(f"-w {drift_window_threshold}")
        cli.append(f"-z {switch_significance}")
        cli.append(f"-a {adwin_delta}")
        cli.append(f"-M {max_size}")
        cli.append(f"-e {memory_estimate_period}")
        cli.append(f"-k {tebst_digits}")
        if numerical_multiway:
            cli.append("-N")
        cli.append(f"-R {numerical_multiway_radius}")
        cli.append(f"-x {max_depth if max_depth is not None else 0}")
        if no_bootstrap_sampling:
            cli.append("-b")
        if binary_split:
            cli.append("-B")
        if stop_mem_management:
            cli.append("-s")
        if remove_poor_attrs:
            cli.append("-p")
        if no_pre_prune:
            cli.append("-u")
        if nominal_attributes:
            cli.append(f"-n {','.join(str(i) for i in nominal_attributes)}")

        super().__init__(
            schema=schema,
            CLI=" ".join(cli),
            random_seed=random_seed,
            moa_learner=_MOA_HATR(),
        )
