import os
import sys
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'implementation')))

from capymoa.instance import RegressionInstance
from river import tree as river_tree, drift as river_drift, preprocessing as river_prep
from hatr_capymoa import HoeffdingAdaptiveTreeRegressor as CapyHATR


# Costruttori

def build_river(p):
    """Costruisce un River HATR dal dict di parametri p."""
    return river_tree.HoeffdingAdaptiveTreeRegressor(
        grace_period=p["grace_period"],
        delta=p["delta"],
        tau=p["tau"],
        leaf_prediction=p["leaf_prediction"],
        bootstrap_sampling=p["bootstrap_sampling"],
        drift_window_threshold=p["drift_window_threshold"],
        switch_significance=p["switch_significance"],
        min_samples_split=p["min_samples_split"],
        model_selector_decay=p["model_selector_decay"],
        drift_detector=river_drift.ADWIN(delta=p["adwin_delta"], clock=p["adwin_clock"]),
        seed=0,
    )


def build_capymoa(p, schema):
    """Costruisce un CapyMOA HATR dal dict di parametri p."""
    return CapyHATR(
        schema,
        grace_period=p["grace_period"],
        split_confidence=p["delta"],
        tie_threshold=p["tau"],
        leaf_prediction=p["leaf_prediction"],
        bootstrap_sampling=p["bootstrap_sampling"],
        drift_window_threshold=p["drift_window_threshold"],
        switch_significance=p["switch_significance"],
        min_samples_split=p["min_samples_split"],
        model_selector_decay=p["model_selector_decay"],
        adwin_delta=p["adwin_delta"],
        random_seed=0,
    )


def build_capymoa_mem(p, schema, max_size_mb: float = 500.0,
                      memory_estimate_period: int = 1_000_000):
    """Come build_capymoa, con parametri di memory management configurabili."""
    return CapyHATR(
        schema,
        grace_period=p["grace_period"],
        split_confidence=p["delta"],
        tie_threshold=p["tau"],
        leaf_prediction=p["leaf_prediction"],
        bootstrap_sampling=p["bootstrap_sampling"],
        drift_window_threshold=p["drift_window_threshold"],
        switch_significance=p["switch_significance"],
        min_samples_split=p["min_samples_split"],
        model_selector_decay=p["model_selector_decay"],
        adwin_delta=p["adwin_delta"],
        max_size_mb=max_size_mb,
        memory_estimate_period=memory_estimate_period,
        random_seed=0,
    )


def build_capymoa_mp(p, schema, merit_preprune: bool):
    """Come build_capymoa, ma con merit_preprune configurabile."""
    return CapyHATR(
        schema,
        grace_period=p["grace_period"],
        split_confidence=p["delta"],
        tie_threshold=p["tau"],
        leaf_prediction=p["leaf_prediction"],
        bootstrap_sampling=p["bootstrap_sampling"],
        drift_window_threshold=p["drift_window_threshold"],
        switch_significance=p["switch_significance"],
        min_samples_split=p["min_samples_split"],
        model_selector_decay=p["model_selector_decay"],
        adwin_delta=p["adwin_delta"],
        merit_preprune=merit_preprune,
        random_seed=0,
    )


# Runner prequential

def _feat_names(X, feature_names):
    if feature_names is not None:
        return feature_names
    return [f"x{j}" for j in range(X.shape[1])]


def run_river(model, X, y, use_scaler=False, feature_names=None):
    """Valutazione prequential River (predici-poi-allena). Restituisce (preds, cum_mae)."""
    names = _feat_names(X, feature_names)
    scaler = river_prep.StandardScaler() if use_scaler else None
    preds, cum_mae_vals = [], []
    mae_sum = 0.0
    for i in range(len(X)):
        x = {f: float(X[i, j]) for j, f in enumerate(names)}
        xp = scaler.transform_one(x) if scaler else x
        pred = model.predict_one(xp) or 0.0
        preds.append(pred)
        mae_sum += abs(y[i] - pred)
        cum_mae_vals.append(mae_sum / (i + 1))
        if scaler:
            scaler.learn_one(x)
            xl = scaler.transform_one(x)
        else:
            xl = x
        model.learn_one(xl, float(y[i]))
    return np.array(preds), np.array(cum_mae_vals)


def run_capymoa(model, X, y, schema, use_scaler=False, feature_names=None):
    """Valutazione prequential CapyMOA (predici-poi-allena). Restituisce (preds, cum_mae)."""
    names = _feat_names(X, feature_names)
    scaler = river_prep.StandardScaler() if use_scaler else None
    preds, cum_mae_vals = [], []
    mae_sum = 0.0
    for i in range(len(X)):
        if scaler:
            x = {f: float(X[i, j]) for j, f in enumerate(names)}
            xs = scaler.transform_one(x)
            arr_pred = np.array([xs[f] for f in names], dtype=float)
        else:
            arr_pred = X[i]
        pred = model.predict(RegressionInstance(schema, (arr_pred, float(y[i])))) or 0.0
        preds.append(pred)
        mae_sum += abs(y[i] - pred)
        cum_mae_vals.append(mae_sum / (i + 1))
        if scaler:
            scaler.learn_one(x)
            xs2 = scaler.transform_one(x)
            arr_learn = np.array([xs2[f] for f in names], dtype=float)
        else:
            arr_learn = X[i]
        model.train(RegressionInstance(schema, (arr_learn, float(y[i]))))
    return np.array(preds), np.array(cum_mae_vals)


# Helper struttura albero

def moa_struct(jh):
    """Restituisce (nodes, leaves) da un oggetto Java HATR."""
    m = {mm.getName(): mm.getValue() for mm in jh.getModelMeasurements()}
    return int(m["tree nodes"]), int(m["tree leaves"])


def get_moa_struct(learner):
    """Restituisce dict {nodes, leaves, active, inactive, alt} da un CapyMOA learner."""
    m = {mm.getName(): mm.getValue() for mm in learner.moa_learner.getModelMeasurements()}
    return {
        "nodes":    int(m.get("tree nodes", 0)),
        "leaves":   int(m.get("tree leaves", 0)),
        "active":   int(m.get("active leaves", 0)),
        "inactive": int(m.get("inactive leaves", 0)),
        "alt":      int(m.get("alternate trees", 0)),
    }


def get_river_struct(model):
    """Restituisce dict {nodes, leaves, active, inactive, alt} da un modello River."""
    s = model.summary
    return {
        "nodes":    s.get("n_nodes", 0),
        "leaves":   s.get("n_leaves", 0),
        "active":   s.get("n_active_leaves", 0),
        "inactive": s.get("n_inactive_leaves", 0),
        "alt":      s.get("n_alternate_trees", 0),
    }
