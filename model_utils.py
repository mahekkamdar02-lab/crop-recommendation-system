"""Shared helpers: feature list, model loading, top-k prediction, SHAP explanation."""
import joblib
import numpy as np
import pandas as pd

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_bundle(path="models/crop_model.joblib"):
    """Bundle = dict(model, label_encoder, features, metrics, model_name)."""
    return joblib.load(path)


def top_k(bundle, row: pd.DataFrame, k=3):
    """Return [(crop, probability), ...] for the k most likely crops."""
    proba = bundle["model"].predict_proba(row[bundle["features"]])[0]
    idx = np.argsort(proba)[::-1][:k]
    crops = bundle["label_encoder"].inverse_transform(idx)
    return [(c, float(proba[i])) for c, i in zip(crops, idx)], idx


def explain(bundle, row: pd.DataFrame, class_index: int):
    """SHAP value per feature for one class. Returns a Series, or None if SHAP is missing."""
    try:
        import shap
    except ImportError:
        return None
    X = row[bundle["features"]]
    sv = shap.TreeExplainer(bundle["model"]).shap_values(X)
    # shap returns a list (old versions) or an (n, features, classes) array (new versions)
    if isinstance(sv, list):
        vals = np.asarray(sv[class_index])[0]
    else:
        sv = np.asarray(sv)
        vals = sv[0, :, class_index] if sv.ndim == 3 else sv[0]
    return pd.Series(vals, index=bundle["features"])


def global_importance(bundle):
    m = bundle["model"]
    if hasattr(m, "feature_importances_"):
        return pd.Series(m.feature_importances_, index=bundle["features"])
    return None
