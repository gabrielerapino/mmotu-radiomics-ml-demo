"""Researcher-facing HTML report with saved figures."""
from __future__ import annotations
import json
from html import escape
from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"mmotu-radiomics-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from .data import CLASS_NAMES


def make_figures(labels, cv_results, final_results, outdir, synthetic=False):
    outdir.mkdir(parents=True, exist_ok=True)
    class_name = lambda c: f"Synthetic class {c}" if synthetic else CLASS_NAMES.get(int(c), str(c))
    names = list(cv_results)
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [cv_results[n]["macro_f1"]["mean"] for n in names]
    errs = [cv_results[n]["macro_f1"]["std"] for n in names]
    ax.bar(names, vals, yerr=errs, color="#427AA1", capsize=4); ax.set_ylim(0,1)
    ax.set_ylabel("Macro F1 (mean ± SD, 5 folds)"); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(outdir/"cv_model_comparison.png", dpi=150); plt.close(fig)
    for name, result in final_results.items():
        classes = result["classes"]
        display_names = [class_name(c) for c in classes]
        cm = np.asarray(result["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7,6))
        ConfusionMatrixDisplay(cm, display_labels=display_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"{name}: {'synthetic holdout' if synthetic else 'official validation set'}"); fig.tight_layout()
        fig.savefig(outdir/f"confusion_{name.lower().replace(' ', '_')}.png",dpi=140); plt.close(fig)
        probs = np.asarray(result["probabilities"]); truth = np.asarray(labels)
        y_bin=label_binarize(truth,classes=classes)
        if y_bin.shape[1]==1: y_bin=np.column_stack([1-y_bin[:,0],y_bin[:,0]])
        fig, ax = plt.subplots(figsize=(6,5))
        for j,c in enumerate(classes):
            if j>=y_bin.shape[1] or len(np.unique(y_bin[:,j]))<2: continue
            fpr,tpr,_=roc_curve(y_bin[:,j],probs[:,j])
            ax.plot(fpr,tpr,label=f"{class_name(c)} (AUC {auc(fpr,tpr):.2f})",alpha=.8)
        ax.plot([0,1],[0,1],"--",color="gray"); ax.set(xlabel="False positive rate",ylabel="True positive rate",title=f"{name}: one-vs-rest ROC")
        ax.legend(fontsize=7,loc="lower right"); fig.tight_layout()
        fig.savefig(outdir/f"roc_{name.lower().replace(' ', '_')}.png",dpi=140); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],"--",color="gray",label="Perfect calibration")
        for j,c in enumerate(classes):
            binary = (truth == c).astype(int)
            if len(np.unique(binary)) < 2: continue
            frac, mean = calibration_curve(binary, probs[:,j], n_bins=6, strategy="quantile")
            ax.plot(mean,frac,marker="o",label=class_name(c))
        ax.set(xlabel="Mean predicted probability",ylabel="Observed frequency",title=f"{name}: one-vs-rest calibration")
        ax.legend(fontsize=7,loc="best"); fig.tight_layout()
        fig.savefig(outdir/f"calibration_{name.lower().replace(' ', '_')}.png",dpi=140); plt.close(fig)


def generate_report(audit, feature_info, cv_results, final_results, valid_labels, outdir, seed):
    synthetic = audit.get("synthetic_data",False)
    figs=outdir/"figures"; make_figures(valid_labels,cv_results,final_results,figs,synthetic)
    rows=[]
    for name,res in cv_results.items():
        gap_note = "Review" if res["accuracy_gap_mean"] >= 0.15 else "No large mean gap flagged"
        rows.append(f"<tr><td>{name}</td><td>{res['accuracy']['mean']:.3f} ± {res['accuracy']['std']:.3f}</td>"
                    f"<td>{res['macro_f1']['mean']:.3f} ± {res['macro_f1']['std']:.3f}</td>"
                    f"<td>{res['balanced_accuracy']['mean']:.3f} ± {res['balanced_accuracy']['std']:.3f}</td>"
                    f"<td>{res['train_accuracy_mean']:.3f}</td><td>{res['accuracy_gap_mean']:.3f} ({gap_note})</td></tr>")
    model_sections=[]
    for name,res in final_results.items():
        p=figs/f"confusion_{name.lower().replace(' ','_')}.png"
        r=figs/f"roc_{name.lower().replace(' ','_')}.png"
        c=figs/f"calibration_{name.lower().replace(' ','_')}.png"
        recall_text="; ".join(f"{('Synthetic class ' + k) if synthetic else CLASS_NAMES.get(int(k),k)}: {v:.2f}" for k,v in res["per_class_recall"].items())
        model_sections.append(f"<h3>{name}</h3><p>Accuracy {res['accuracy']:.3f}; macro F1 {res['macro_f1']:.3f}; balanced accuracy {res['balanced_accuracy']:.3f}; macro OVR AUC {res['macro_ovr_auc'] if res['macro_ovr_auc'] is not None else 'not estimable'}; multiclass Brier score {res['multiclass_brier']:.3f}; log loss {res['log_loss']:.3f}. Per-class recall: {recall_text}.</p><img src='figures/{p.name}'><img src='figures/{r.name}'><img src='figures/{c.name}'>")
    dist="; ".join(f"{k}: {v}" for k,v in audit["class_distribution"].items())
    download_note = audit.get("public_dataset_download_status",{}).get("reason",audit.get("fallback_reason",""))
    banner = "<strong>Synthetic fallback data:</strong> these results are only a software demonstration and contain no patient observations." if synthetic else ""
    validation_strategy = ("Five-fold duplicate-grouped image-level CV and a stratified internal holdout were used only to demonstrate the pipeline on synthetic arrays. There is no official validation partition in this fallback." if synthetic else
                          "Five-fold duplicate-grouped image-level cross-validation on the official training partition is the primary estimate. The original stratified image-level CV is retained as a comparator. The official validation partition is held out; exact pixel duplicates that also occur in training are removed for the duplicate-clean estimate.")
    research_question = ("Does the example code run from generated image arrays and masks through feature extraction and model evaluation?" if synthetic else
                         "Can a conservative set of radiomic measurements from annotated regions distinguish the official MMOTU image categories in a small, leakage-aware research experiment?")
    target_note = ("Generated class codes are software fixtures; they do not represent MMOTU categories or clinical outcomes." if synthetic else
                   "The official 8-class labels are retained; no benign/malignant mapping is imposed.")
    limitations = ("Only synthetic arrays were used because complete MMOTU images, masks, and label lists were not available locally. The scores measure software behavior on generated patterns only and provide no evidence about ultrasound data, patients, or clinical use." if synthetic else
                   "Image-level splits can place images from the same person in different folds. The public labels do not supply usable patient identifiers. Results are from one source and one acquisition setting, and can be optimistic due to shared patient/image acquisition characteristics. Radiomic features may be affected by image display, compression, and annotation choices. These results do not prove clinical validity, diagnostic utility, external generalization, or patient-level performance.")
    html=f"""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>MMOTU radiomics research demo</title>
<style>
body{{font:16px/1.5 system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#18212b}}
img{{max-width:100%;height:auto;margin:1rem 1rem 1rem 0}}
table{{border-collapse:collapse;max-width:100%}}td,th{{padding:.5rem;border:1px solid #bbb;text-align:left;vertical-align:top}}.note{{background:#edf4f8;padding:1rem}}
.export-bar{{display:flex;align-items:center;gap:.75rem;margin:0 0 1.5rem}}
.export-bar button{{border:0;border-radius:.4rem;background:#245b78;color:white;padding:.65rem 1rem;font:inherit;font-weight:600;cursor:pointer}}
.export-bar button:hover{{background:#19445c}}.export-hint{{font-size:.9rem;color:#4d5963}}
@media print{{
  @page{{size:A4;margin:16mm}}
  html,body{{max-width:none;margin:0;padding:0;color:#18212b;background:white;font-size:10pt}}
  .export-bar{{display:none}}
  h1,h2,h3{{break-after:avoid-page;page-break-after:avoid}}
  p,li{{orphans:3;widows:3}}
  table,img{{break-inside:avoid;page-break-inside:avoid}}
  img{{max-width:100%;margin:.5rem 0;print-color-adjust:exact;-webkit-print-color-adjust:exact}}
  .note{{print-color-adjust:exact;-webkit-print-color-adjust:exact}}
}}
</style>
<div class='export-bar'><button type='button' onclick='window.print()'>Export PDF</button><span class='export-hint'>Choose “Save as PDF” in the print dialog.</span></div>
<h1>Radiomics from ovarian ultrasound: research demonstration</h1><p class='note'>{banner} This is not a clinical diagnostic tool and does not establish clinical validity.</p>
<h2>1. Research question</h2><p>{research_question}</p>
<h2>2. Dataset and data quality</h2><p>{audit['dataset']}: {audit['n_label_rows']} labeled rows; {audit.get('usable_labeled_rows',audit['n_label_rows'])} rows usable for modeling; {audit.get('excluded_rows_from_modeling',0)} rows excluded for missing or invalid files and duplicate filenames. Class distribution: {dist}. Images found: {audit['n_images_found']}; masks found: {audit['n_masks_found']}; missing image labels: {len(audit['missing_images_for_labeled_rows'])}; missing masks: {len(audit['missing_masks_for_labeled_rows'])}; empty masks: {len(audit['empty_masks'])}; dimension/read errors: {len(audit['image_mask_mismatches'])}; duplicate label filenames: {len(audit['duplicate_filenames_in_label_lists'])}; image names with differing file contents: {len(audit.get('duplicate_image_filenames_on_disk',[]))}; identical image copies across sources: {len(audit.get('identical_duplicate_image_filenames',[]))}.</p><p>Data access note: {download_note}</p>
<p>Patient grouping available: <b>{audit['patient_grouping_available']}</b>. {audit['patient_grouping_note']}</p>
<h2>3. Validation strategy</h2><p>{validation_strategy} No patient-level independence can be claimed because no reliable patient IDs are available. Random seed: {seed}.</p>
<h2>4. Radiomic features</h2><p>{feature_info['engine']}. {len(feature_info['feature_names'])} feature columns. Physical pixel spacing is unavailable, so shape quantities are in pixels and have no physical interpretation.</p>
<h2>5. Models and CV results</h2><p>Dummy prior baseline, logistic regression, random forest, and XGBoost where available. Imputation, variance filtering, scaling, and mutual-information feature selection are fitted inside each fold's training pipeline. A mean accuracy gap of at least 0.15 is flagged for review.</p><table><tr><th>Model</th><th>CV accuracy</th><th>Macro F1</th><th>Balanced accuracy</th><th>Train accuracy</th><th>Mean train–validation accuracy gap</th></tr>{''.join(rows)}</table><img src='figures/cv_model_comparison.png'>
<h2>6. {'Synthetic holdout' if synthetic else 'Official validation results'}, confusion matrices, ROC curves, and calibration</h2>{''.join(model_sections)}
<h2>7. Methodological safeguards</h2><ul><li>Feature selection, imputation, and scaling are learned only from each training fold.</li><li>No tuning uses the held-out evaluation partition.</li><li>Random seed is recorded ({seed}); folds are stratified by image class.</li><li>No patient identifiers or clinical covariates are fabricated.</li><li>No physical spacing is assumed.</li><li>{target_note}</li></ul>
<h2>8. Limitations and what this result does not prove</h2><p>{limitations}</p>
<p>Full machine-readable audit, metrics, and feature settings are saved alongside this report.</p></html>"""
    (outdir/"demo_report.html").write_text(html,encoding="utf-8")


def append_duplicate_audit(report_path, leakage_audit, comparison, conclusion):
    """Add the duplicate leakage comparison while preserving the existing report."""
    def value(row, key):
        v = row.get(key)
        return "—" if v is None or pd.isna(v) else f"{float(v):.3f}"

    headers = ["Model", "Original accuracy", "Grouped accuracy", "Original macro F1",
               "Grouped macro F1", "Original balanced acc.", "Grouped balanced acc.",
               "Original OVR AUC", "Grouped OVR AUC", "Original train acc. / gap",
               "Grouped train acc. / gap"]
    header_html = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body_rows = []
    for _, row in comparison.iterrows():
        body_rows.append(
            "<tr><th>{}</th><td>{} ± {}</td><td>{} ± {}</td><td>{} ± {}</td><td>{} ± {}</td>"
            "<td>{} ± {}</td><td>{} ± {}</td><td>{} ± {}</td><td>{} ± {}</td>"
            "<td>{} / {}</td><td>{} / {}</td></tr>".format(
                escape(str(row["model"])),
                value(row, "original_cv_accuracy_mean"), value(row, "original_cv_accuracy_std"),
                value(row, "duplicate_safe_cv_accuracy_mean"), value(row, "duplicate_safe_cv_accuracy_std"),
                value(row, "original_cv_macro_f1_mean"), value(row, "original_cv_macro_f1_std"),
                value(row, "duplicate_safe_cv_macro_f1_mean"), value(row, "duplicate_safe_cv_macro_f1_std"),
                value(row, "original_cv_balanced_accuracy_mean"), value(row, "original_cv_balanced_accuracy_std"),
                value(row, "duplicate_safe_cv_balanced_accuracy_mean"), value(row, "duplicate_safe_cv_balanced_accuracy_std"),
                value(row, "original_cv_macro_ovr_auc_mean"), value(row, "original_cv_macro_ovr_auc_std"),
                value(row, "duplicate_safe_cv_macro_ovr_auc_mean"), value(row, "duplicate_safe_cv_macro_ovr_auc_std"),
                value(row, "original_cv_train_accuracy_mean"), value(row, "original_cv_train_validation_gap_mean"),
                value(row, "duplicate_safe_cv_train_accuracy_mean"), value(row, "duplicate_safe_cv_train_validation_gap_mean"),
            )
        )

    dist_before = "; ".join(f"{escape(k)}: {v}" for k, v in leakage_audit["official_validation_class_distribution_before"].items())
    dist_after = "; ".join(f"{escape(k)}: {v}" for k, v in leakage_audit["official_validation_class_distribution_after"].items())
    validation_scope = escape(leakage_audit.get("validation_scope", "official validation"))
    boundary_scope = ("official train/validation boundary" if leakage_audit.get("validation_scope") == "official validation"
                      else "internal train/validation boundary")
    partition_crossers = (leakage_audit["n_duplicate_groups_spanning_official_train_validation"]
                          + leakage_audit.get("n_duplicate_groups_spanning_internal_train_validation", 0))
    holdout_rows = []
    for _, row in comparison.iterrows():
        holdout_rows.append("<tr><th>{}</th><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
                            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(str(row["model"])),
            value(row, "original_official_validation_accuracy"), value(row, "duplicate_clean_official_validation_accuracy"),
            value(row, "original_official_validation_macro_f1"), value(row, "duplicate_clean_official_validation_macro_f1"),
            value(row, "original_official_validation_balanced_accuracy"), value(row, "duplicate_clean_official_validation_balanced_accuracy"),
            value(row, "original_official_validation_macro_ovr_auc"), value(row, "duplicate_clean_official_validation_macro_ovr_auc"),
        ))

    section = f"""<section id='duplicate-leakage-audit'>
<h2>Duplicate leakage audit</h2>
<p>Grouping uses SHA256 of the grayscale decoded pixel array (including its shape and dtype); file-byte SHA256 is stored separately. The inventory covers {leakage_audit['n_source_image_files_audited']} physical source files for {leakage_audit['n_usable_images']} usable labeled images. It found {leakage_audit['n_duplicate_groups']} exact duplicate source-file clusters containing {leakage_audit['n_images_in_duplicate_groups']} files ({leakage_audit['n_unique_pixel_groups']} unique pixel-content groups total). These copies map to {leakage_audit['n_model_observations_in_duplicate_groups']} labeled observations, each used once by the loader; {leakage_audit['n_duplicate_groups_with_multiple_model_observations']} pixel groups contain multiple modeling observations. {leakage_audit['n_duplicate_groups_spanning_original_cv_folds']} duplicate groups crossed folds under the previous stratified image-level CV. {partition_crossers} groups crossed the {boundary_scope}.</p>
<p>The corrected method is <strong>duplicate-grouped image-level cross-validation</strong> using StratifiedGroupKFold. It leaves {leakage_audit['n_duplicate_groups_spanning_duplicate_safe_cv_folds']} duplicate groups crossing validation folds. Patient IDs are unavailable; no patient grouping is claimed.</p>
<p><strong>{escape(conclusion)}</strong></p>
<div style='overflow-x:auto'><table class='duplicate-cv'><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>
<p>{validation_scope.capitalize()} cleaning: {leakage_audit['original_validation_image_count']} images originally; {leakage_audit['contaminated_validation_images_removed']} removed because their pixel group also occurs in training; {leakage_audit['cleaned_validation_image_count']} remain. Class distribution before: {dist_before}. After: {dist_after}.</p>
<table><thead><tr><th>Model</th><th>Original official accuracy</th><th>Duplicate-clean accuracy</th><th>Original macro F1</th><th>Duplicate-clean macro F1</th><th>Original balanced accuracy</th><th>Duplicate-clean balanced accuracy</th><th>Original OVR AUC</th><th>Duplicate-clean OVR AUC</th></tr></thead><tbody>{''.join(holdout_rows)}</tbody></table>
<style>.duplicate-cv{{font-size:.78rem;white-space:nowrap}}#duplicate-leakage-audit table{{border-collapse:collapse;margin:1rem 0}}#duplicate-leakage-audit th,#duplicate-leakage-audit td{{padding:.4rem;border:1px solid #bbb;text-align:right}}#duplicate-leakage-audit th:first-child{{text-align:left}}</style>
</section>"""
    report_path = Path(report_path)
    html = report_path.read_text(encoding="utf-8")
    marker = "<h2>1. Research question</h2>"
    if marker not in html:
        raise ValueError("Could not locate the research question section in the existing report")
    html = html.replace(marker, section + marker, 1)
    report_path.write_text(html, encoding="utf-8")
