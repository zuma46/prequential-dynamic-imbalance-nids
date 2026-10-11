from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


class StreamClassifier:
    name: str
    deterministic: bool

    def predict_one(self, x: dict[str, float], fallback: str) -> str:
        raise NotImplementedError

    def learn_one(self, x: dict[str, float], y: str, sampling_lambda: float = 1.0) -> None:
        raise NotImplementedError


class RiverSingleClassifier(StreamClassifier):
    __slots__ = ("name", "deterministic", "learner")

    def __init__(self, name: str, learner: object, deterministic: bool = True) -> None:
        self.name = name
        self.deterministic = deterministic
        self.learner = learner

    def predict_one(self, x: dict[str, float], fallback: str) -> str:
        prediction = self.learner.predict_one(x)
        return fallback if prediction is None else str(prediction)

    def learn_one(self, x: dict[str, float], y: str, sampling_lambda: float = 1.0) -> None:
        # Single baseline learner: exactly one update per observed instance.
        self.learner.learn_one(x, y)


class OnlineBaggingClassifier(StreamClassifier):
    """Literature-faithful Online Bagging / MOOB / MUOB ensemble.

    Fidelity decisions:
    1. Ensemble prediction is hard majority voting over base-classifier labels.
    2. For each base learner, K ~ Poisson(lambda) is sampled independently.
    3. The current instance is used for K *sequential* learner updates.
       A single weighted update is intentionally NOT used because it is not
       guaranteed to be state-equivalent for incremental Hoeffding Trees.

    The implementation keeps the hot path small and uses one NumPy vectorized
    Poisson draw per instance for all ensemble members.
    """

    __slots__ = (
        "name",
        "deterministic",
        "learners",
        "rng",
        "class_order",
        "n_models",
    )

    def __init__(
        self,
        name: str,
        learner_factory: Callable[[], object],
        n_models: int,
        seed: int,
        class_order: Iterable[str],
    ) -> None:
        n_models = int(n_models)
        if n_models < 1:
            raise ValueError("n_models must be >= 1")

        self.name = name
        self.deterministic = False
        self.learners = [learner_factory() for _ in range(n_models)]
        self.rng = np.random.default_rng(int(seed))
        self.class_order = tuple(str(label) for label in class_order)
        self.n_models = n_models

    def predict_one(self, x: dict[str, float], fallback: str) -> str:
        # Hard majority voting, matching the MOOB/MUOB experimental protocol.
        votes: defaultdict[str, int] = defaultdict(int)

        for model in self.learners:
            prediction = model.predict_one(x)
            if prediction is not None:
                votes[str(prediction)] += 1

        if not votes:
            return fallback

        # Deterministic tie-breaking by the fixed class order. With an odd
        # ensemble size ties are uncommon, but deterministic handling is still
        # required for reproducibility.
        best_label = fallback
        best_votes = -1

        for label in self.class_order:
            count = votes.get(label, 0)
            if count > best_votes:
                best_votes = count
                best_label = label

        # Defensive fallback for a label unexpectedly absent from class_order.
        if best_votes <= 0:
            return max(votes.items(), key=lambda item: (item[1], item[0]))[0]

        return best_label

    def learn_one(self, x: dict[str, float], y: str, sampling_lambda: float = 1.0) -> None:
        lam = max(0.0, float(sampling_lambda))
        if lam == 0.0:
            return

        # One vectorized RNG call is faster than one RNG call per base learner.
        multiplicities = self.rng.poisson(lam=lam, size=self.n_models)
        learners = self.learners

        # Exact Poisson-repeat semantics from Online Bagging / MOOB / MUOB.
        # Local bindings reduce Python attribute lookup inside the repeat loop.
        for i in range(self.n_models):
            k = int(multiplicities[i])
            if k <= 0:
                continue

            learn_one = learners[i].learn_one
            for _ in range(k):
                learn_one(x, y)


@dataclass(slots=True)
class SamplingRates:
    moob: float
    muob: float


class DynamicClassSizeTracker:
    __slots__ = ("class_to_index", "sizes", "theta", "floor", "lambda_cap")

    def __init__(
        self,
        class_names: list[str],
        theta: float = 0.9,
        floor: float = 1e-12,
        lambda_cap: float | None = None,
    ) -> None:
        if not class_names:
            raise ValueError("class_names cannot be empty")
        self.class_to_index = {label: i for i, label in enumerate(class_names)}
        self.sizes = np.full(len(class_names), 1.0 / len(class_names), dtype=np.float64)
        self.theta = float(theta)
        self.floor = float(floor)
        self.lambda_cap = None if lambda_cap is None else float(lambda_cap)

    def update_and_get_rates(self, y: str) -> SamplingRates:
        idx = self.class_to_index[y]

        # Time-decayed class-size update:
        # w_k(t) = theta*w_k(t-1) + (1-theta)*I(y_t == c_k)
        self.sizes *= self.theta
        self.sizes[idx] += 1.0 - self.theta

        # Numerical protection only. With theta=0.9, the currently observed
        # class receives at least 0.1 before the MOOB/MUOB ratio is computed.
        np.maximum(self.sizes, self.floor, out=self.sizes)
        self.sizes /= self.sizes.sum()

        current = float(self.sizes[idx])
        moob = float(self.sizes.max() / current)
        muob = float(self.sizes.min() / current)

        # MOOB in the final fidelity configuration uses no cap. The optional
        # field is retained only for backward compatibility with P1 outputs.
        if self.lambda_cap is not None:
            moob = min(moob, self.lambda_cap)

        return SamplingRates(moob=moob, muob=muob)


def make_ht_factory(ht_cfg: dict, seed: int | None = None) -> Callable[[], object]:
    def factory() -> object:
        from river import tree

        kwargs = dict(ht_cfg)
        signature = inspect.signature(tree.HoeffdingTreeClassifier)
        if "seed" in signature.parameters and seed is not None:
            kwargs["seed"] = int(seed)
        valid = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return tree.HoeffdingTreeClassifier(**valid)

    return factory


def build_models(
    cfg: dict,
    class_names: list[str],
    seed: int,
) -> tuple[list[StreamClassifier], DynamicClassSizeTracker | None]:
    algorithms = list(cfg["models"]["algorithms"])
    ensemble_size = int(cfg["models"]["ensemble_size"])
    ht_cfg = dict(cfg["models"]["ht"])
    models: list[StreamClassifier] = []
    deterministic_seed = int(cfg["experiment"]["deterministic_seed"])

    # Stable algorithm-specific offsets make random streams invariant to
    # whether algorithms are run together or as separate HPC processes.
    algorithm_seed_offsets = {
        "HT": 0,
        "OB": 1_000_003,
        "MOOB": 2_000_006,
        "MOOB_L20": 2_000_006,  # backward-compatible P1 alias only
        "MUOB": 3_000_009,
    }

    for algorithm in algorithms:
        model_seed = int(seed) + algorithm_seed_offsets[algorithm]
        factory = make_ht_factory(ht_cfg, seed=model_seed)

        if algorithm == "HT":
            if int(seed) == deterministic_seed:
                models.append(RiverSingleClassifier("HT", factory(), deterministic=True))

        elif algorithm == "OB":
            models.append(
                OnlineBaggingClassifier("OB", factory, ensemble_size, model_seed, class_names)
            )

        elif algorithm == "MOOB":
            models.append(
                OnlineBaggingClassifier("MOOB", factory, ensemble_size, model_seed, class_names)
            )

        elif algorithm == "MOOB_L20":
            # Retained solely so old P1 configs/checkpoints remain readable.
            models.append(
                OnlineBaggingClassifier("MOOB_L20", factory, ensemble_size, model_seed, class_names)
            )

        elif algorithm == "MUOB":
            models.append(
                OnlineBaggingClassifier("MUOB", factory, ensemble_size, model_seed, class_names)
            )

        else:
            raise ValueError(algorithm)

    needs_tracker = any(model.name.startswith("MOOB") or model.name == "MUOB" for model in models)
    tracker = None

    if needs_tracker:
        cap = cfg["models"].get("moob_lambda_cap")
        if "MOOB" in algorithms and "MOOB_L20" not in algorithms:
            # Literature-faithful MOOB: no lambda cap.
            cap = None

        tracker = DynamicClassSizeTracker(
            class_names=class_names,
            theta=float(cfg["models"]["class_size_theta"]),
            floor=float(cfg["models"]["class_size_floor"]),
            lambda_cap=cap,
        )

    return models, tracker
