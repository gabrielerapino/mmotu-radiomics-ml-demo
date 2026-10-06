"""Evaluation summaries and calibration helpers."""
from __future__ import annotations
import numpy as np
from sklearn.calibration import calibration_curve


def multiclass_calibration(y, probabilities, classes, n_bins=8):
    rows = []
    y = np.asarray(y); probabilities = np.asarray(probabilities)
    for index, cls in enumerate(classes):
        truth = (y == cls).astype(int)
        if truth.min() == truth.max(): continue
        observed, predicted = calibration_curve(truth, probabilities[:, index], n_bins=n_bins,
                                                 strategy="quantile")
        rows.append({"class": int(cls), "predicted": predicted.tolist(), "observed": observed.tolist()})
    return rows
