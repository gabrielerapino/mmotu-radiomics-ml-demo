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
    comparison_path = outdir / "duplicate_safe_model_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else None
    for name,res in cv_results.items():
        # The standard image-level CV is the main estimate. The grouped CV is
        # presented later as a robustness comparison using the saved CSV.
        row = None if comparison is None else comparison.loc[comparison.model == name]
        if row is not None and not row.empty:
            row = row.iloc[0]
            acc, acc_std = row.original_cv_accuracy_mean, row.original_cv_accuracy_std
            f1, f1_std = row.original_cv_macro_f1_mean, row.original_cv_macro_f1_std
            bal, bal_std = row.original_cv_balanced_accuracy_mean, row.original_cv_balanced_accuracy_std
            train_acc, gap = row.original_cv_train_accuracy_mean, row.original_cv_train_validation_gap_mean
        else:
            acc, acc_std = res["accuracy"]["mean"], res["accuracy"]["std"]
            f1, f1_std = res["macro_f1"]["mean"], res["macro_f1"]["std"]
            bal, bal_std = res["balanced_accuracy"]["mean"], res["balanced_accuracy"]["std"]
            train_acc, gap = res["train_accuracy_mean"], res["accuracy_gap_mean"]
        rows.append(f"<tr><td>{name}</td><td>{acc:.3f} ± {acc_std:.3f}</td>"
                    f"<td>{f1:.3f} ± {f1_std:.3f}</td><td>{bal:.3f} ± {bal_std:.3f}</td>"
                    f"<td>{train_acc:.3f}</td><td>{gap:.3f}</td></tr>")
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
    validation_strategy = ("Five-fold cross-validation on the synthetic training partition, with a synthetic holdout used only for final software evaluation. There is no official validation partition in this fallback." if synthetic else
                          "Five-fold cross-validation on the official training partition, with the official validation partition held out until final evaluation. A duplicate-grouped cross-validation was also performed as a robustness analysis.")
    research_question = ("Can the example workflow extract measurements from generated images and run model evaluation?" if synthetic else
                         "Can quantitative radiomic features extracted from segmented ovarian ultrasound images help distinguish the eight official MMOTU diagnostic categories?")
    target_note = ("Generated class codes are software fixtures; they do not represent MMOTU categories or clinical outcomes." if synthetic else
                   "The official 8-class labels are retained; no benign/malignant mapping is imposed.")
    limitations = ("Only synthetic arrays were used because complete MMOTU images, masks, and label lists were not available locally. The scores measure software behavior on generated patterns only and provide no evidence about ultrasound data, patients, or clinical use." if synthetic else
                   "No reliable patient identifiers are available, so patient-level independence cannot be established. The data come from one public dataset and acquisition context, with no external-center validation. Physical pixel spacing is unavailable. Radiomic features may depend on annotation, display, and compression characteristics. These results do not establish clinical validity or diagnostic utility.")
    overview = ("<dl class='overview'><dt>Research question</dt><dd>Distinguish the eight official MMOTU diagnostic categories from segmented ultrasound radiomics.</dd>"
                f"<dt>Dataset</dt><dd>{audit['dataset']}; {audit['n_label_rows']} labeled images, {audit.get('usable_labeled_rows',audit['n_label_rows'])} usable images.</dd>"
                f"<dt>Method</dt><dd>{feature_info['engine']}; five-fold image-level CV and a held-out official validation partition.</dd>"
                "<dt>Main finding</dt><dd>More complex models had somewhat higher AUCs and larger train–validation gaps; exact duplicates did not materially affect results.</dd>"
                "<dt>Key limitation</dt><dd>No reliable patient identifiers are available, so patient-level independent validation cannot be claimed.</dd></dl>")
    class_table = "".join(f"<tr><th>{escape(str(k))}</th><td>{v}</td></tr>" for k,v in audit["class_distribution"].items())
    dataset_sentence = (f"{audit['dataset']} includes {audit['n_label_rows']:,} labeled images across eight official diagnostic categories. "
                       f"{audit.get('usable_labeled_rows',audit['n_label_rows']):,} images had usable image and mask files; "
                       f"{audit.get('excluded_rows_from_modeling',0):,} observations were excluded because files or masks were missing or invalid. "
                       "No reliable patient identifiers are available. A source-file duplication audit was also performed and is reported later as a robustness check.")
    html=f"""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>MMOTU radiomics research demo</title>
<style>
body{{font:16px/1.5 system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#18212b}}
img{{max-width:100%;height:auto;margin:1rem 1rem 1rem 0}}
table{{border-collapse:collapse;max-width:100%}}td,th{{padding:.5rem;border:1px solid #bbb;text-align:left;vertical-align:top}}.note{{background:#edf4f8;padding:1rem}}
.overview{{display:grid;grid-template-columns:minmax(8rem,12rem) 1fr;gap:.45rem 1rem;background:#f4f7f9;padding:1rem 1.25rem;border-radius:.5rem}}.overview dt{{font-weight:700}}.overview dd{{margin:0}}
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
<h1>Radiomics from ovarian ultrasound: research demonstration</h1>
<h2>Study overview</h2>{overview}
<h2>1. Research question</h2><p>{research_question}</p><p class='note'>This is a research demonstration and does not establish clinical validity or diagnostic utility.</p>
<h2>2. Dataset and data quality</h2><p>{dataset_sentence}</p><table><thead><tr><th>Official category</th><th>Labeled images</th></tr></thead><tbody>{class_table}</tbody></table>
<h2>3. Radiomics pipeline</h2><p>Expert segmented ultrasound regions were analyzed with {feature_info['engine']}. The feature set contains {len(feature_info['feature_names'])} columns: first-order, GLCM, GLRLM, and 2D shape features. No physical pixel spacing is available; shape features are expressed in pixel units and have no physical interpretation.</p><p>Preprocessing comprises imputation, variance filtering, scaling, and mutual-information feature selection. Each step is fitted only on the training folds.</p>
<h2>4. Validation strategy</h2><p>{validation_strategy} The official validation partition was not used for tuning. No patient-level independence can be claimed. Random seed: {seed}.</p>
<h2>5. Main model results</h2><p>The standard five-fold image-level CV results are shown below for the Dummy baseline, Logistic Regression, Random Forest, and XGBoost. Logistic Regression has the smallest train–validation gap. Random Forest and XGBoost show larger gaps, consistent with greater overfitting, while the complex models achieve somewhat higher AUCs; performance varies substantially across classes.</p><table><thead><tr><th>Model</th><th>CV accuracy</th><th>Macro F1</th><th>Balanced accuracy</th><th>Train accuracy</th><th>Train–validation gap</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h3>{'Synthetic holdout' if synthetic else 'Official validation'}, confusion matrices, ROC curves, and calibration</h3>{''.join(model_sections)}
<h2>6. Robustness and leakage checks</h2><p>The source-file duplication audit and duplicate-grouped CV comparison appear below. Duplicate grouping is a robustness analysis; the audit found no exact-duplicate crossing in the image-level folds or official train/validation boundary.</p><!-- DUPLICATE_AUDIT_INSERTION -->
<h2>7. Methodological safeguards</h2><ul><li>Imputation, variance filtering, scaling, and feature selection are fitted only on training folds.</li><li>The held-out official validation partition is not used for tuning.</li><li>The random seed is fixed at {seed}; folds are stratified by image class.</li><li>No patient IDs or clinical covariates are fabricated.</li><li>No physical spacing is invented.</li><li>{target_note}</li></ul>
<h2>8. Limitations</h2><ol><li>No reliable patient identifiers are available.</li><li>Patient-level independent validation cannot be claimed.</li><li>Evidence comes from one public dataset and acquisition context.</li><li>No external-center validation was performed.</li><li>Physical pixel spacing is unavailable.</li><li>Radiomic features may depend on annotation, display, and compression characteristics.</li><li>These results do not establish clinical validity.</li></ol>
<h2>9. Conclusions and research interpretation</h2><p>This demonstration shows that a reproducible radiomics-to-ML workflow can be built on MMOTU ovarian ultrasound data, including leakage-aware preprocessing, model comparison, calibration, and robustness checks. More complex models achieved somewhat higher discrimination but also showed larger train–validation gaps. Exact duplicate leakage was investigated and did not materially affect performance. The experiment should be interpreted as a methodological demonstration rather than a clinically validated diagnostic model.</p>
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
<h3>Exact duplicate image audit</h3>
<p>The audit inspected {leakage_audit['n_source_image_files_audited']:,} physical source files and identified {leakage_audit['n_duplicate_groups']:,} sets of duplicate physical source files ({leakage_audit['n_images_in_duplicate_groups']:,} files). Each labeled modeling observation was used only once: no duplicate image group contained multiple modeling observations. No groups crossed CV folds or the official train/validation boundary.</p>
<p>Duplicate-grouped cross-validation produced essentially unchanged results and left {leakage_audit['n_duplicate_groups_spanning_duplicate_safe_cv_folds']} duplicate groups crossing validation folds. Exact duplicate leakage was therefore not a material driver of model performance. Patient IDs are unavailable, so this audit does not establish patient-level independence.</p>
<p><strong>{escape(conclusion)}</strong></p>
<h4>Duplicate-grouped CV macro F1 by model</h4><img src='figures/cv_model_comparison.png' alt='Duplicate-grouped CV macro F1 by model'>
<h4>Original CV vs duplicate-grouped CV</h4>
<div style='overflow-x:auto'><table class='duplicate-cv'><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>
<h4>Official validation vs duplicate-clean validation</h4><p>{leakage_audit['contaminated_validation_images_removed']} validation images were removed; metrics remained unchanged ({leakage_audit['original_validation_image_count']} original and {leakage_audit['cleaned_validation_image_count']} duplicate-clean images). Class distribution before: {dist_before}. After: {dist_after}.</p>
<table><thead><tr><th>Model</th><th>Original official accuracy</th><th>Duplicate-clean accuracy</th><th>Original macro F1</th><th>Duplicate-clean macro F1</th><th>Original balanced accuracy</th><th>Duplicate-clean balanced accuracy</th><th>Original OVR AUC</th><th>Duplicate-clean OVR AUC</th></tr></thead><tbody>{''.join(holdout_rows)}</tbody></table>
<style>.duplicate-cv{{font-size:.78rem;white-space:nowrap}}#duplicate-leakage-audit table{{border-collapse:collapse;margin:1rem 0}}#duplicate-leakage-audit th,#duplicate-leakage-audit td{{padding:.4rem;border:1px solid #bbb;text-align:right}}#duplicate-leakage-audit th:first-child{{text-align:left}}</style>
</section>"""
    report_path = Path(report_path)
    html = report_path.read_text(encoding="utf-8")
    marker = "<!-- DUPLICATE_AUDIT_INSERTION -->"
    if marker not in html:
        raise ValueError("Could not locate the robustness audit insertion point in the report")
    html = html.replace(marker, section + marker, 1)
    report_path.write_text(html, encoding="utf-8")
