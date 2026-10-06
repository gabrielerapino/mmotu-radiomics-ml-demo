"""End-to-end experiment orchestration used by the CLI and notebook."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .data import load_mmotu, make_synthetic_dataset
from .duplicates import build_duplicate_groups, clean_validation_indices
from .duplicate_analysis import make_duplicate_leakage_audit, build_model_comparison, leakage_conclusion
from .radiomics import extract_features, save_features
from .pipeline import run_cv, fit_final_models, make_cv_splits
from .report import generate_report, append_duplicate_audit


def run_experiment(data_root=Path("data/raw/OTU_2d"), output_dir=Path("outputs"),
                   seed=42, n_splits=5, synthetic_samples_per_class=12,
                   selection_percentile=25, force_synthetic=False):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    images=masks=None
    try:
        if force_synthetic: raise FileNotFoundError("Synthetic fallback requested")
        labels,audit=load_mmotu(Path(data_root))
        if labels.image_path.isna().any() or labels.mask_path.isna().any():
            raise FileNotFoundError("MMOTU labels were found, but not all matching images and masks are present")
    except (FileNotFoundError, ValueError) as exc:
        labels,audit,images,masks=make_synthetic_dataset(synthetic_samples_per_class,seed)
        status_path=Path(data_root)/"download_status.json"
        download_status=json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
        audit["fallback_reason"]=str(exc)
        if download_status and not download_status.get("success",False):
            audit["public_dataset_download_status"]=download_status
    (output_dir/"data_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    duplicate_groups = build_duplicate_groups(labels, Path(data_root), images=images)
    duplicate_groups.to_csv(output_dir/"duplicate_groups.csv", index=False)
    modeling_duplicate_groups = duplicate_groups.loc[
        duplicate_groups.is_modeling_observation
    ].reset_index(drop=True)
    if audit.get("synthetic_data"):
        cv_base_indices=np.arange(len(labels))
        final_train_idx, final_valid_idx = train_test_split(
            cv_base_indices, test_size=0.2, stratify=labels.label.to_numpy(), random_state=seed
        )
        validation_scope="synthetic internal holdout"
    else:
        cv_base_indices=np.flatnonzero(labels.official_split.to_numpy()=="train")
        final_train_idx=cv_base_indices.copy()
        final_valid_idx=np.flatnonzero(labels.official_split.to_numpy()=="validation")
        validation_scope="official validation"
    train_labels=labels.iloc[cv_base_indices].reset_index(drop=True)
    train_images=[images[i] for i in cv_base_indices] if images is not None else None
    train_masks=[masks[i] for i in cv_base_indices] if masks is not None else None
    features,feature_info=extract_features(train_labels,train_images,train_masks)
    save_features(features,train_labels,output_dir/"radiomics_features.csv")
    (output_dir/"radiomics_config.json").write_text(json.dumps(feature_info,indent=2),encoding="utf-8")
    X=features.select_dtypes(include=[np.number]); y=train_labels.label.to_numpy()
    cv_group_ids=modeling_duplicate_groups.iloc[cv_base_indices].duplicate_group_id.to_numpy()
    original_splits=make_cv_splits(y,seed=seed,n_splits=n_splits)
    corrected_splits=make_cv_splits(y,seed=seed,n_splits=n_splits,groups=cv_group_ids)
    original_cv_results,original_selected=run_cv(X,y,list(X.columns),seed,n_splits,selection_percentile)
    grouped_cv_results,grouped_selected=run_cv(
        X,y,list(X.columns),seed,n_splits,selection_percentile,groups=cv_group_ids
    )
    original_selected.to_csv(output_dir/"selected_features_image_level_original.csv",index=False)
    grouped_selected.to_csv(output_dir/"selected_features_by_fold.csv",index=False)
    grouped_selected.to_csv(output_dir/"selected_features_duplicate_safe_by_fold.csv",index=False)
    if audit.get("synthetic_data"):
        X_train,y_train=X.iloc[final_train_idx],labels.iloc[final_train_idx].label.to_numpy()
        X_valid,y_valid=X.iloc[final_valid_idx],labels.iloc[final_valid_idx].label.to_numpy()
    else:
        valid_labels=labels.iloc[final_valid_idx].reset_index(drop=True)
        valid_images=None; valid_masks=None
        valid_features,valid_info=extract_features(valid_labels,valid_images,valid_masks)
        X_train,X_valid=X,valid_features.reindex(columns=X.columns)
        y_train,y_valid=y,valid_labels.label.to_numpy()
    clean_valid_idx=clean_validation_indices(
        modeling_duplicate_groups.iloc[final_train_idx].duplicate_group_id,
        modeling_duplicate_groups.iloc[final_valid_idx].duplicate_group_id,
    )
    clean_X_valid=X_valid.iloc[clean_valid_idx]
    clean_y_valid=y_valid[clean_valid_idx]
    final_results,_=fit_final_models(X_train,y_train,X_valid,y_valid,list(X.columns),seed,selection_percentile)
    clean_final_results=({} if len(clean_y_valid)==0 else fit_final_models(
        X_train,y_train,clean_X_valid,clean_y_valid,list(X.columns),seed,selection_percentile
    )[0])

    duplicate_audit=make_duplicate_leakage_audit(
        duplicate_groups,
        original_train_indices=cv_base_indices,
        original_splits=original_splits,
        corrected_splits=corrected_splits,
        train_indices=final_train_idx,
        validation_indices=final_valid_idx,
        clean_validation_indices_=clean_valid_idx,
        validation_scope=validation_scope,
    )
    (output_dir/"duplicate_leakage_audit.json").write_text(
        json.dumps(duplicate_audit,indent=2),encoding="utf-8"
    )
    comparison=build_model_comparison(original_cv_results,grouped_cv_results,final_results,clean_final_results)
    comparison.to_csv(output_dir/"duplicate_safe_model_comparison.csv",index=False)
    duplicate_validation={
        "validation_scope":validation_scope,
        "counts":{
            "original_validation_images":int(len(final_valid_idx)),
            "contaminated_validation_images_removed":int(duplicate_audit["contaminated_validation_images_removed"]),
            "cleaned_validation_images":int(len(clean_y_valid)),
        },
        "class_distribution_before":duplicate_audit["official_validation_class_distribution_before"],
        "class_distribution_after":duplicate_audit["official_validation_class_distribution_after"],
        "original_validation":final_results,
        "duplicate_clean_validation":clean_final_results,
    }
    (output_dir/"duplicate_safe_validation.json").write_text(
        json.dumps(duplicate_validation,indent=2),encoding="utf-8"
    )
    (output_dir/"metrics.json").write_text(json.dumps({
        "cv":grouped_cv_results,
        "official_validation":final_results,
        "cv_image_level_original":original_cv_results,
        "cv_duplicate_grouped":grouped_cv_results,
        "official_validation_original":final_results,
        "official_validation_duplicate_clean":clean_final_results,
        "feature_info":feature_info,"random_seed":seed,"cv_folds":n_splits,
    },indent=2),encoding="utf-8")
    generate_report(audit,feature_info,grouped_cv_results,final_results,y_valid,output_dir,seed)
    conclusion=leakage_conclusion(
        comparison,
        groups_crossing_original_folds=duplicate_audit["n_duplicate_groups_spanning_original_cv_folds"],
    )
    append_duplicate_audit(output_dir/"demo_report.html",duplicate_audit,comparison,conclusion)
    print(f"Dataset: {audit['dataset']} | labeled rows: {audit['n_label_rows']} | features: {features.shape[1]}")
    print(f"CV folds: {n_splits} | duplicate groups: {duplicate_audit['n_duplicate_groups']} | "
          f"cross-fold duplicate clusters: {duplicate_audit['n_duplicate_groups_spanning_original_cv_folds']} | "
          f"report: {output_dir/'demo_report.html'}")
    if audit.get("synthetic_data"):
        print("WARNING: using synthetic fallback data; results are not from patients or MMOTU.")
    return {"labels":labels,"features":features,"feature_info":feature_info,"audit":audit,
            "cv_results":grouped_cv_results,"original_cv_results":original_cv_results,
            "duplicate_safe_cv_results":grouped_cv_results,
            "final_results":final_results,"duplicate_clean_validation_results":clean_final_results,
            "duplicate_leakage_audit":duplicate_audit,"output_dir":output_dir}
