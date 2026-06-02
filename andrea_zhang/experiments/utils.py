import os
import sys
import numpy as np
from time import perf_counter

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'implementation')))

from capymoa.stream import Schema
from capymoa.instance import RegressionInstance
from river import tree as river_tree, drift as river_drift, preprocessing as river_prep
from hatr_capymoa import HoeffdingAdaptiveTreeRegressor as CapyHATR

try:
    from pympler import asizeof as _asizeof
    _PYMPLER_OK = True
except ImportError:
    _PYMPLER_OK = False

try:
    from moa.core import SizeOf as _MoaSizeOf
    _SIZEOF_OK = True
except Exception:
    _SIZEOF_OK = False

# Dataset 

def materialize_stream(stream, max_instances=None):
    """Estrae (X, y, schema) da uno stream CapyMOA."""
    schema = stream.get_schema()
    X, y = [], []
    try:
        stream.restart()
    except Exception:
        pass
    n = 0
    while stream.has_more_instances():
        if max_instances is not None and n >= max_instances:
            break
        inst = stream.next_instance()
        X.append(np.asarray(inst.x, dtype=float))
        y.append(float(inst.y_value))
        n += 1
    return np.asarray(X), np.asarray(y), schema


def make_synthetic(n=10000, drift_at=5000, seed=42):
    """Dataset sintetico con concept drift a drift_at."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    noise = 0.1 * rng.randn(n)
    y = np.where(
        np.arange(n) < drift_at,
        2*X[:, 0] + 3*X[:, 1] - X[:, 2] + noise,
        -1.5*X[:, 0] + 0.5*X[:, 3]*X[:, 4] + 5.0 + noise,
    )
    schema = Schema.from_custom(
        features=[f"x{i}" for i in range(5)] + ["y"],
        target="y", categories=None, name="synthetic_drift")
    return X, y, schema


# Helpers interni ai runner

def river_dict(row):
    return {f"x{j}": float(row[j]) for j in range(len(row))}


def scaled_array(scaler, x_dict, nfeat):
    s = scaler.transform_one(x_dict)
    return np.fromiter((s[f"x{j}"] for j in range(nfeat)), dtype=float, count=nfeat)


# Costruttori

def build_river(cfg, base):
    """Costruisce un River HATR dalla configurazione cfg e dai parametri base."""
    return river_tree.HoeffdingAdaptiveTreeRegressor(
        grace_period=base["grace_period"],
        delta=base["delta"],
        tau=base["tau"],
        leaf_prediction=cfg["leaf_prediction"],
        bootstrap_sampling=cfg["bootstrap_sampling"],
        model_selector_decay=base["model_selector_decay"],
        drift_window_threshold=base["drift_window_threshold"],
        switch_significance=base["switch_significance"],
        min_samples_split=base["min_samples_split"],
        drift_detector=river_drift.ADWIN(delta=base["adwin_delta"], clock=32),
        seed=base["seed"],
    )


def build_capy(cfg, schema, base):
    """Costruisce un CapyMOA HATR dalla configurazione cfg e dai parametri base."""
    return CapyHATR(
        schema,
        grace_period=base["grace_period"],
        split_confidence=base["delta"],
        tie_threshold=base["tau"],
        leaf_prediction=cfg["leaf_prediction"],
        bootstrap_sampling=cfg["bootstrap_sampling"],
        model_selector_decay=base["model_selector_decay"],
        drift_window_threshold=base["drift_window_threshold"],
        switch_significance=base["switch_significance"],
        min_samples_split=base["min_samples_split"],
        adwin_delta=base["adwin_delta"],
        learning_ratio=base["learning_ratio"],
        random_seed=base["seed"],
    )


# Runner prequential

def prequential_river(X, y, cfg, base, scale=True):
    """Valutazione prequential (predici-poi-allena) con River. Restituisce (preds, model)."""
    model = build_river(cfg, base)
    scaler = river_prep.StandardScaler() if scale else None
    preds = np.empty(len(X))
    for i in range(len(X)):
        x = river_dict(X[i])
        xp = scaler.transform_one(x) if scaler else x
        preds[i] = model.predict_one(xp) or 0.0
        if scaler:
            scaler.learn_one(x)
            xl = scaler.transform_one(x)
        else:
            xl = x
        model.learn_one(xl, float(y[i]))
    return preds, model


def prequential_capy(X, y, schema, cfg, base, scale=True):
    """Valutazione prequential (predici-poi-allena) con CapyMOA. Restituisce (preds, learner)."""
    learner = build_capy(cfg, schema, base)
    scaler = river_prep.StandardScaler() if scale else None
    nfeat = X.shape[1]
    preds = np.empty(len(X))
    for i in range(len(X)):
        if scaler:
            x = river_dict(X[i])
            arr = scaled_array(scaler, x, nfeat)
        else:
            arr = X[i]
        preds[i] = learner.predict(RegressionInstance(schema, (arr, float(y[i])))) or 0.0
        if scaler:
            scaler.learn_one(x)
            arr2 = scaled_array(scaler, x, nfeat)
        else:
            arr2 = X[i]
        learner.train(RegressionInstance(schema, (arr2, float(y[i]))))
    return preds, learner


# Runner cronometrati

def timed_prequential_river(X, y, cfg, base, scale=True):
    """Come prequential_river, ma restituisce (t_pred, t_train, model)."""
    model = build_river(cfg, base)
    scaler = river_prep.StandardScaler() if scale else None
    t_pred = t_train = 0.0
    for i in range(len(X)):
        x = river_dict(X[i])
        xp = scaler.transform_one(x) if scaler else x
        t0 = perf_counter(); _ = model.predict_one(xp) or 0.0; t_pred += perf_counter() - t0
        if scaler:
            scaler.learn_one(x)
            xl = scaler.transform_one(x)
        else:
            xl = x
        t0 = perf_counter(); model.learn_one(xl, float(y[i])); t_train += perf_counter() - t0
    return t_pred, t_train, model


def timed_prequential_capy(X, y, schema, cfg, base, scale=True):
    """Come prequential_capy, ma restituisce (t_pred, t_train, learner)."""
    learner = build_capy(cfg, schema, base)
    scaler = river_prep.StandardScaler() if scale else None
    nfeat = X.shape[1]
    t_pred = t_train = 0.0
    for i in range(len(X)):
        if scaler:
            x = river_dict(X[i])
            arr = scaled_array(scaler, x, nfeat)
        else:
            arr = X[i]
        inst_p = RegressionInstance(schema, (arr, float(y[i])))
        _ = inst_p.java_instance  # forza conversione numpy→Java prima del timer
        t0 = perf_counter(); _ = learner.predict(inst_p) or 0.0; t_pred += perf_counter() - t0
        if scaler:
            scaler.learn_one(x)
            arr2 = scaled_array(scaler, x, nfeat)
        else:
            arr2 = X[i]
        inst_t = RegressionInstance(schema, (arr2, float(y[i])))
        _ = inst_t.java_instance  # forza conversione numpy→Java prima del timer
        t0 = perf_counter(); learner.train(inst_t); t_train += perf_counter() - t0
    return t_pred, t_train, learner


# Dimensione modello

def river_model_mb(model):
    """Dimensione in MB dell'oggetto modello River (richiede pympler)."""
    if not _PYMPLER_OK:
        return float("nan")
    return _asizeof.asizeof(model) / 1e6


def capy_model_mb(learner):
    """Dimensione in MB dell'oggetto modello CapyMOA (richiede SizeOf agent)."""
    if not _SIZEOF_OK:
        return float("nan")
    size = _MoaSizeOf.fullSizeOf(learner.moa_learner)
    return size / 1e6 if size > 0 else float("nan")


# Runner con campionamento a intervalli

def traced_prequential_river(X, y, cfg, base, scale=True, sample_every=500):
    """Prequential River con snapshot periodici di model_MB, mae, rmse. Restituisce lista di dict."""
    model = build_river(cfg, base)
    scaler = river_prep.StandardScaler() if scale else None
    trace = []
    w_abs, w_sq = [], []
    for i in range(len(X)):
        x = river_dict(X[i])
        xp = scaler.transform_one(x) if scaler else x
        pred = model.predict_one(xp) or 0.0
        err = y[i] - pred
        w_abs.append(abs(err)); w_sq.append(err ** 2)
        if scaler:
            scaler.learn_one(x)
            xl = scaler.transform_one(x)
        else:
            xl = x
        model.learn_one(xl, float(y[i]))
        if (i + 1) % sample_every == 0:
            win = w_abs[-sample_every:]; wsq = w_sq[-sample_every:]
            trace.append(dict(n=i + 1,
                              model_MB=river_model_mb(model),
                              mae=float(np.mean(win)),
                              rmse=float(np.sqrt(np.mean(wsq)))))
    return trace


def traced_prequential_capy(X, y, schema, cfg, base, scale=True, sample_every=500):
    """Prequential CapyMOA con snapshot periodici di model_MB, mae, rmse. Restituisce lista di dict."""
    learner = build_capy(cfg, schema, base)
    scaler = river_prep.StandardScaler() if scale else None
    nfeat = X.shape[1]
    trace = []
    w_abs, w_sq = [], [] # errori assoluti e quadratici
    for i in range(len(X)):
        if scaler:
            x = river_dict(X[i])
            arr = scaled_array(scaler, x, nfeat)
        else:
            arr = X[i]
        inst_p = RegressionInstance(schema, (arr, float(y[i])))
        pred = learner.predict(inst_p) or 0.0
        err = y[i] - pred
        w_abs.append(abs(err)); w_sq.append(err ** 2)
        if scaler:
            scaler.learn_one(x)
            arr2 = scaled_array(scaler, x, nfeat)
        else:
            arr2 = X[i]
        learner.train(RegressionInstance(schema, (arr2, float(y[i]))))
        if (i + 1) % sample_every == 0:
            win = w_abs[-sample_every:]; wsq = w_sq[-sample_every:]
            trace.append(dict(n=i + 1,
                              model_MB=capy_model_mb(learner),
                              mae=float(np.mean(win)),
                              rmse=float(np.sqrt(np.mean(wsq)))))
    return trace


# Metriche

def mae(yt, yp):
    return float(np.mean(np.abs(yt - yp)))

def rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def r2(yt, yp):
    ss = np.sum((yt - yp) ** 2)
    st = np.sum((yt - np.mean(yt)) ** 2)
    return float(1 - ss / st) if st > 0 else 0.0

def cum_mae(yt, yp):
    return np.cumsum(np.abs(yt - yp)) / np.arange(1, len(yt) + 1)

def cum_rmse(yt, yp):
    return np.sqrt(np.cumsum((yt - yp) ** 2) / np.arange(1, len(yt) + 1))

def win_mae(yt, yp, w=500):
    return np.convolve(np.abs(yt - yp), np.ones(w) / w, mode="valid")

def cum_r2(yt, yp):
    r2_arr = np.empty(len(yt))
    for i in range(1, len(yt) + 1):
        ss = np.sum((yt[:i] - yp[:i]) ** 2)
        st = np.sum((yt[:i] - np.mean(yt[:i])) ** 2)
        r2_arr[i - 1] = 1 - ss / st if st > 0 else 0.0
    return r2_arr
