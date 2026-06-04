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


def make_fried_drift(n=40000, drift_at=20000, seed=42):
    """Friedman con drift abrupt a drift_at: relazione cambia da x0-x4 a x5-x9."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n, 10))
    noise = rng.randn(n)
    y_pre  = (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2
              + 10*X[:,3] + 5*X[:,4] + noise)
    y_post = (10*np.sin(np.pi*X[:,5]*X[:,6]) + 20*(X[:,7]-0.5)**2
              + 10*X[:,8] + 5*X[:,9] + noise)
    y = np.where(np.arange(n) < drift_at, y_pre, y_post)
    schema = Schema.from_custom(
        features=[f"x{i}" for i in range(10)] + ["y"],
        target="y", categories=None, name="fried_drift")
    return X, y, schema


def make_rbf_regression(n=30000, drift_at=15000, n_centers=5, n_features=10, seed=42):
    """RBF regression con drift graduale: i centroidi si spostano dopo drift_at."""
    rng = np.random.RandomState(seed)
    centers_a = rng.randn(n_centers, n_features)
    centers_b = centers_a + 0.5 * rng.randn(n_centers, n_features)
    weights   = rng.randn(n_centers, n_features)
    X = rng.randn(n, n_features)
    y = np.empty(n)
    for i in range(n):
        t = max(0.0, (i - drift_at) / (n - drift_at)) if i >= drift_at else 0.0
        curr = (1 - t) * centers_a + t * centers_b
        closest = int(np.argmin(np.sum((X[i] - curr) ** 2, axis=1)))
        y[i] = float(weights[closest] @ X[i]) + 0.1 * rng.randn()
    schema = Schema.from_custom(
        features=[f"x{i}" for i in range(n_features)] + ["y"],
        target="y", categories=None, name="rbf_regression")
    return X, y, schema


def load_electricity_demand(path, max_instances=None):
    """Carica electricity.arff usando nswdemand come target di regressione."""
    from scipy.io import arff
    import pandas as pd
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    # day è categorico ({1..7}) — scipy restituisce byte string, decode esplicito
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    feat_cols = ['date', 'day', 'period', 'nswprice', 'vicprice', 'vicdemand', 'transfer']
    X = df[feat_cols].astype(float).values
    y = df['nswdemand'].astype(float).values
    if max_instances:
        X, y = X[:max_instances], y[:max_instances]
    schema = Schema.from_custom(
        features=feat_cols + ['nswdemand'],
        target='nswdemand', categories=None, name='electricity_demand')
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

def prequential_eval(X, y, framework, cfg, base,
                     schema=None, scale=True,
                     time_algo=False, sample_every=None):
    """Valutazione prequential unificata per River e CapyMOA.

    framework   : "river" | "capy"
    schema      : richiesto se framework="capy"
    time_algo   : misura t_predict e t_train escludendo conversione JNI (capy)
    sample_every: campiona model_MB/mae/rmse ogni N istanze

    Restituisce (preds, model, t_predict, t_train, trace)
      t_predict = t_train = 0.0  se time_algo=False
      trace = []                 se sample_every=None
    """
    is_capy = framework == "capy"
    if is_capy and schema is None:
        raise ValueError("schema richiesto per framework='capy'")

    model  = build_capy(cfg, schema, base) if is_capy else build_river(cfg, base)
    scaler = river_prep.StandardScaler() if scale else None
    nfeat  = X.shape[1]
    n      = len(X)
    preds  = np.empty(n)
    t_predict = t_train = 0.0
    trace  = []
    w_abs, w_sq = [], []

    for i in range(n):
        x_dict = river_dict(X[i])

        # predict
        if is_capy:
            arr    = scaled_array(scaler, x_dict, nfeat) if scaler else X[i]
            inst_p = RegressionInstance(schema, (arr, float(y[i])))
            if time_algo:
                _ = inst_p.java_instance  # forza conversione numpy->Java prima del timer
                t0 = perf_counter()
            pred = model.predict(inst_p) or 0.0
            if time_algo:
                t_predict += perf_counter() - t0
        else:
            xp   = scaler.transform_one(x_dict) if scaler else x_dict
            if time_algo:
                t0 = perf_counter()
            pred = model.predict_one(xp) or 0.0
            if time_algo:
                t_predict += perf_counter() - t0
        preds[i] = pred

        if scaler:
            scaler.learn_one(x_dict)

        # train — riusa arr/xp già calcolati prima dell'aggiornamento dello scaler
        if is_capy:
            inst_t = RegressionInstance(schema, (arr, float(y[i])))
            if time_algo:
                _ = inst_t.java_instance  # forza conversione numpy->Java prima del timer
                t0 = perf_counter()
            model.train(inst_t)
            if time_algo:
                t_train += perf_counter() - t0
        else:
            if time_algo:
                t0 = perf_counter()
            model.learn_one(xp, float(y[i]))
            if time_algo:
                t_train += perf_counter() - t0

        if sample_every:
            w_abs.append(abs(y[i] - pred))
            w_sq.append((y[i] - pred) ** 2)
            if (i + 1) % sample_every == 0:
                mb = capy_model_mb(model) if is_capy else river_model_mb(model)
                trace.append(dict(
                    n        = i + 1,
                    model_MB = mb,
                    mae      = float(np.mean(w_abs[-sample_every:])),
                    rmse     = float(np.sqrt(np.mean(w_sq[-sample_every:]))),
                ))

    return preds, model, t_predict, t_train, trace


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

def win_rmse(yt, yp, w=500):
    return np.sqrt(np.convolve((yt - yp) ** 2, np.ones(w) / w, mode="valid"))

def cum_r2(yt, yp):
    r2_arr = np.empty(len(yt))
    for i in range(1, len(yt) + 1):
        ss = np.sum((yt[:i] - yp[:i]) ** 2)
        st = np.sum((yt[:i] - np.mean(yt[:i])) ** 2)
        r2_arr[i - 1] = 1 - ss / st if st > 0 else 0.0
    return r2_arr
