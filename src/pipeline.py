"""Leakage-safe stratified cross-validation and held-out official validation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from functools import partial
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, log_loss, roc_auc_score)

MODEL_NAMES = ["Dummy", "Logistic Regression", "Random Forest", "XGBoost"]


def make_models(seed=42):
    models = {"Dummy": DummyClassifier(strategy="prior"),
              "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", C=1.0),
              "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                                       class_weight="balanced_subsample", random_state=seed,
                                                       n_jobs=1)}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.05,
                                          subsample=0.85, colsample_bytree=0.85,
                                          objective="multi:softprob", eval_metric="mlogloss",
                                          random_state=seed, n_jobs=1, verbosity=0)
    except ImportError:
        models["XGBoost"] = None
    return models


def make_pipeline(model, seed=42, selection_percentile=25):
    steps = [("imputer", SimpleImputer(strategy="median")),
             ("variance", VarianceThreshold()),
             ("scaler", StandardScaler()),
             ("selector", SelectPercentile(partial(mutual_info_classif, random_state=seed), percentile=selection_percentile))]
    if model.__class__.__name__ != "DummyClassifier":
        steps.append(("model", model))
    else:
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    return Pipeline(steps)


def metric_dict(y, pred, proba, classes):
    proba = np.asarray(proba, dtype=float)
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sums, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sums > 0)
    one_hot = label_binarize(y, classes=classes)
    if one_hot.shape[1] == 1:
        one_hot = np.column_stack([1-one_hot[:,0], one_hot[:,0]])
    out = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
           "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
           "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
           "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
           "log_loss": log_loss(y, proba, labels=classes),
           "multiclass_brier": float(np.mean(np.sum((proba-one_hot)**2, axis=1))),
           "per_class_recall": {str(int(cls)): float(recall_score(np.asarray(y)==cls, np.asarray(pred)==cls, zero_division=0)) for cls in classes},
           "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist()}
    try:
        out["macro_ovr_auc"] = roc_auc_score(y, proba, labels=classes, multi_class="ovr", average="macro")
    except ValueError:
        out["macro_ovr_auc"] = None
    return out


def make_cv_splits(y, seed=42, n_splits=5, groups=None):
    """Build deterministic image-level or duplicate-grouped stratified folds."""
    y = np.asarray(y)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y))
    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError("groups must contain one duplicate group ID per observation")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def run_cv(X, y, feature_names, seed=42, n_splits=5, selection_percentile=25, groups=None):
    y = np.asarray(y); classes = np.unique(y)
    splits = make_cv_splits(y, seed=seed, n_splits=n_splits, groups=groups)
    all_results, selected = {}, []
    models = make_models(seed)
    for name, estimator in models.items():
        if estimator is None:
            continue
        fold_metrics, oof = [], np.zeros((len(y), len(classes)))
        train_gaps = []; train_scores = []
        for fold, (train_i, val_i) in enumerate(splits, 1):
            pipe = make_pipeline(clone(estimator), seed, selection_percentile)
            pipe.fit(X.iloc[train_i], y[train_i])
            proba = pipe.predict_proba(X.iloc[val_i])
            model_classes = pipe.named_steps["model"].classes_
            aligned = np.zeros((len(val_i), len(classes)))
            for j, cls in enumerate(model_classes): aligned[:, np.where(classes == cls)[0][0]] = proba[:, j]
            pred = classes[aligned.argmax(axis=1)]
            oof[val_i] = aligned
            fold_metrics.append(metric_dict(y[val_i], pred, aligned, classes))
            train_score = pipe.score(X.iloc[train_i], y[train_i])
            valid_score = accuracy_score(y[val_i], pred)
            train_scores.append(float(train_score))
            train_gaps.append(float(train_score - valid_score))
            selector = pipe.named_steps.get("selector")
            if selector is not None:
                keep = np.array(feature_names)[selector.get_support()]
                selected.extend({"model": name, "fold": fold, "feature": f} for f in keep)
        summary = {}
        keys = [k for k,v in fold_metrics[0].items() if isinstance(v,(float,int)) and v is not None]
        for key in keys:
            vals = [m[key] for m in fold_metrics if m[key] is not None]
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0}
        summary["train_accuracy_mean"] = float(np.mean(train_scores))
        summary["accuracy_gap_mean"] = float(np.mean(train_gaps))
        summary["folds"] = fold_metrics
        summary["oof_probabilities"] = oof.tolist()
        all_results[name] = summary
    return all_results, pd.DataFrame(selected)


def fit_final_models(X_train, y_train, X_valid, y_valid, feature_names, seed=42, selection_percentile=25):
    results, fitted = {}, {}
    for name, estimator in make_models(seed).items():
        if estimator is None: continue
        pipe = make_pipeline(clone(estimator), seed, selection_percentile)
        pipe.fit(X_train, y_train)
        probabilities = pipe.predict_proba(X_valid)
        classes = pipe.named_steps["model"].classes_
        predictions = classes[probabilities.argmax(axis=1)]
        results[name] = metric_dict(y_valid, predictions, probabilities, classes)
        results[name]["predictions"] = predictions.tolist()
        results[name]["probabilities"] = probabilities.tolist()
        results[name]["classes"] = classes.tolist()
        fitted[name] = pipe
    return results, fitted
