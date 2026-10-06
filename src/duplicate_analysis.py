"""Summaries for exact-pixel duplicate leakage and paired model comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CLASS_NAMES
from .duplicates import clean_validation_indices, duplicate_group_sizes


def _class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.label.value_counts().sort_index()
    return {CLASS_NAMES.get(int(label), str(label)): int(count) for label, count in counts.items()}


def make_duplicate_leakage_audit(
    groups: pd.DataFrame,
    original_train_indices: np.ndarray,
    original_splits: list[tuple[np.ndarray, np.ndarray]],
    corrected_splits: list[tuple[np.ndarray, np.ndarray]],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    clean_validation_indices_: np.ndarray,
    validation_scope: str = "official validation",
) -> dict:
    sizes = duplicate_group_sizes(groups)
    duplicate_ids = set(sizes[sizes > 1].index)
    repeated_rows = groups.duplicate_group_id.isin(duplicate_ids)
    model_groups = groups.loc[groups.is_modeling_observation].reset_index(drop=True)
    model_sizes = duplicate_group_sizes(model_groups)
    model_duplicate_ids = set(model_sizes[model_sizes > 1].index)

    train_groups = model_groups.iloc[train_indices].reset_index(drop=True)
    validation_groups = model_groups.iloc[validation_indices].reset_index(drop=True)
    train_group_ids = set(train_groups.duplicate_group_id)
    validation_group_ids = set(validation_groups.duplicate_group_id)
    partition_crossers = train_group_ids & validation_group_ids

    fold_by_row: dict[int, int] = {}
    for fold_number, (_, val_idx) in enumerate(original_splits, start=1):
        for local_index in val_idx:
            fold_by_row[int(local_index)] = fold_number
    fold_spans = []
    cv_training_groups = model_groups.iloc[original_train_indices].reset_index(drop=True)
    for group_id, group_rows in cv_training_groups.groupby("duplicate_group_id", sort=True):
        folds = sorted({fold_by_row[int(i)] for i in group_rows.index if int(i) in fold_by_row})
        if len(group_rows) > 1 and len(folds) > 1:
            fold_spans.append({"duplicate_group_id": group_id, "cv_folds": folds,
                               "filenames": group_rows.filename.astype(str).tolist()})

    corrected_fold_by_row: dict[int, int] = {}
    for fold_number, (_, val_idx) in enumerate(corrected_splits, start=1):
        for local_index in val_idx:
            corrected_fold_by_row[int(local_index)] = fold_number
    corrected_spans = []
    for group_id, group_rows in cv_training_groups.groupby("duplicate_group_id", sort=True):
        folds = sorted({corrected_fold_by_row[int(i)] for i in group_rows.index
                        if int(i) in corrected_fold_by_row})
        if len(group_rows) > 1 and len(folds) > 1:
            corrected_spans.append({"duplicate_group_id": group_id, "cv_folds": folds,
                                    "filenames": group_rows.filename.astype(str).tolist()})

    crossing_details = []
    validation_contaminated = validation_groups.duplicate_group_id.isin(train_group_ids)
    contaminated = validation_groups.loc[validation_contaminated]
    for group_id, group_rows in contaminated.groupby("duplicate_group_id", sort=True):
        train_names = train_groups.loc[train_groups.duplicate_group_id == group_id, "filename"].astype(str).tolist()
        crossing_details.append({"duplicate_group_id": group_id,
                                 "train_filenames": train_names,
                                 "validation_filenames": group_rows.filename.astype(str).tolist()})

    return {
        "duplicate_definition": "SHA256 of grayscale decoded pixel array (shape and dtype included); file-byte SHA256 recorded separately",
        "cv_strategy_original": "StratifiedKFold, shuffled, image-level",
        "cv_strategy_corrected": "duplicate-grouped image-level cross-validation using StratifiedGroupKFold",
        "n_usable_images": int(len(model_groups)),
        "n_unique_pixel_groups": int(len(sizes)),
        "n_duplicate_groups": int(len(duplicate_ids)),
        "n_images_in_duplicate_groups": int(repeated_rows.sum()),
        "n_source_image_files_audited": int(len(groups)),
        "n_model_observations": int(len(model_groups)),
        "n_model_observations_in_duplicate_groups": int(model_groups.duplicate_group_id.isin(duplicate_ids).sum()),
        "n_duplicate_groups_with_multiple_model_observations": int(len(model_duplicate_ids)),
        "n_duplicate_groups_spanning_original_cv_folds": int(len(fold_spans)),
        "duplicate_groups_spanning_original_cv_folds": fold_spans,
        "n_duplicate_groups_spanning_duplicate_safe_cv_folds": int(len(corrected_spans)),
        "duplicate_groups_spanning_duplicate_safe_cv_folds": corrected_spans,
        "validation_scope": validation_scope,
        "n_duplicate_groups_spanning_official_train_validation": int(len(partition_crossers)) if validation_scope == "official validation" else 0,
        "duplicate_groups_spanning_official_train_validation": crossing_details if validation_scope == "official validation" else [],
        "n_duplicate_groups_spanning_internal_train_validation": int(len(partition_crossers)) if validation_scope != "official validation" else 0,
        "duplicate_groups_spanning_internal_train_validation": crossing_details if validation_scope != "official validation" else [],
        "original_validation_image_count": int(len(validation_indices)),
        "contaminated_validation_images_removed": int(len(contaminated)),
        "cleaned_validation_image_count": int(len(clean_validation_indices_)),
        "official_train_class_distribution": _class_distribution(train_groups),
        "official_validation_class_distribution_before": _class_distribution(validation_groups),
        "official_validation_class_distribution_after": _class_distribution(
            validation_groups.iloc[clean_validation_indices_]
        ),
        "patient_grouping_available": False,
        "patient_grouping_note": "No patient IDs are available; duplicate grouping only isolates identical decoded images.",
        "original_cv_fold_image_counts": [int(len(val_idx)) for _, val_idx in original_splits],
        "corrected_cv_group_integrity": "All modeled rows with the same decoded-pixel duplicate_group_id are isolated in one validation fold; no duplicate group is split across validation folds.",
    }


def build_model_comparison(
    original_cv: dict,
    grouped_cv: dict,
    original_validation: dict,
    clean_validation: dict,
) -> pd.DataFrame:
    metric_names = ["accuracy", "macro_f1", "balanced_accuracy", "macro_ovr_auc"]
    rows = []
    for model in original_cv:
        row = {"model": model}
        for prefix, results in (("original_cv", original_cv), ("duplicate_safe_cv", grouped_cv)):
            summary = results[model]
            for metric in metric_names:
                metric_result = summary.get(metric)
                row[f"{prefix}_{metric}_mean"] = None if metric_result is None else metric_result["mean"]
                row[f"{prefix}_{metric}_std"] = None if metric_result is None else metric_result["std"]
            row[f"{prefix}_train_accuracy_mean"] = summary["train_accuracy_mean"]
            row[f"{prefix}_train_validation_gap_mean"] = summary["accuracy_gap_mean"]
        for metric in metric_names:
            old = row[f"original_cv_{metric}_mean"]
            new = row[f"duplicate_safe_cv_{metric}_mean"]
            row[f"duplicate_safe_minus_original_{metric}"] = None if old is None or new is None else new - old
        for prefix, results in (("original_official_validation", original_validation),
                                ("duplicate_clean_official_validation", clean_validation)):
            validation_metrics = results.get(model, {})
            for metric in metric_names:
                row[f"{prefix}_{metric}"] = validation_metrics.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def leakage_conclusion(comparison: pd.DataFrame, material_threshold: float = 0.02,
                       groups_crossing_original_folds: int | None = None) -> str:
    """Derive report language from mean macro-F1 movement across non-dummy models."""
    models = comparison[comparison.model != "Dummy"]
    deltas = models["duplicate_safe_minus_original_macro_f1"].dropna()
    if deltas.empty:
        return "The grouped CV comparison could not estimate macro F1 for enough models to assess duplicate leakage."
    average_change = float(deltas.mean())
    if average_change <= -material_threshold:
        if groups_crossing_original_folds == 0:
            return (f"Duplicate-grouped CV reduced average non-dummy macro F1 by {-average_change:.3f}, "
                    "but no exact-duplicate modeling group crossed the original CV folds; the change cannot be attributed to duplicate leakage.")
        return ("Previous image-level validation was optimistic because identical image content could cross evaluation boundaries. "
                f"Across the three non-dummy models, duplicate-grouped CV reduced mean macro F1 by {-average_change:.3f} "
                f"(material threshold: {material_threshold:.2f}).")
    if abs(average_change) < material_threshold:
        return ("Duplicate-safe splitting produced similar results, suggesting exact duplicate leakage was not a major driver of performance. "
                f"The average non-dummy macro-F1 change was {average_change:+.3f} "
                f"(similarity threshold: ±{material_threshold:.2f}).")
    return (f"Duplicate-safe CV increased average non-dummy macro F1 by {average_change:.3f}; "
            "the original image-level results do not show material inflation from exact duplicates.")
