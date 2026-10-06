You are acting as an autonomous ML engineer building a SMALL BUT REAL clinical-research demo.

Goal:
Build an end-to-end radiomics + tabular ML demo for ovarian/adnexal ultrasound research using the public MMOTU dataset if accessible through normal public means.

This is a research demonstrator, NOT a clinical diagnostic tool.

The success criterion is:
“Given ultrasound images + masks, extract radiomic features, run a leakage-safe ML comparison, and generate a clear research report.”

Do not overbuild.
Do not create a production platform.
Do not build deep-learning segmentation.
Do not spend excessive time on UI.
Do not run huge hyperparameter searches.

==================================================
1. ACCESS / SAFETY RULES
==================================================

Use only normally accessible public resources.

Do NOT:
- bypass authentication;
- bypass Hugging Face gating;
- bypass Google Drive permissions;
- scrape protected pages;
- use restricted patient datasets;
- invent patient IDs or clinical metadata.

If a public dataset download is blocked by a normal access issue:
- document it;
- continue with a small synthetic fallback so the code can run;
- do not spend more than ~10 minutes trying alternative download routes.

==================================================
2. DATASET
==================================================

Primary dataset:
MMOTU / OTU_2d
Repository:
https://github.com/cv516Buaa/MMOTU_DS2Net

First:
- inspect the repository;
- identify the normal public dataset download route;
- inspect labels, masks, filenames, image format, and any patient-level metadata.

Important:
Do NOT assume filenames correspond to patients unless officially documented.

If reliable patient IDs are unavailable:
- use image-level splits;
- clearly state in the report that patient-level independence cannot be guaranteed.

Do not use restricted IOTA raw data.

==================================================
3. PROJECT SCOPE
==================================================

Build a clean Python project with:

- dataset loader;
- image/mask validation;
- radiomic feature extraction;
- preprocessing;
- leakage-safe feature selection;
- Logistic Regression;
- Random Forest;
- XGBoost;
- 5-fold cross-validation;
- final model comparison;
- calibration;
- HTML report;
- README;
- basic automated tests.

No Streamlit in this first version unless it is trivial after everything else works.

==================================================
4. PROJECT STRUCTURE
==================================================

Use a simple structure such as:

project/
  README.md
  requirements.txt
  config.yaml

  src/
    data.py
    radiomics.py
    pipeline.py
    models.py
    evaluation.py
    report.py

  tests/
    test_splits.py
    test_preprocessing.py
    test_feature_selection.py
    test_reproducibility.py

  outputs/

  run_demo.py

Keep modules understandable and reasonably small.

==================================================
5. DATA AUDIT
==================================================

Before modeling, inspect and report:

- number of images;
- number of masks;
- missing masks;
- class distribution;
- duplicate filenames;
- image dimensions;
- mask dimensions;
- empty masks;
- image/mask mismatches;
- available metadata;
- whether patient grouping is available.

Save:
outputs/data_audit.json

Also print a concise summary.

==================================================
6. RADIOMICS
==================================================

Use PyRadiomics if practical.

Extract a conservative set of features:
- first-order;
- texture;
- shape only if valid for the available data.

Do NOT invent physical pixel spacing.

If physical spacing is unavailable:
- state clearly that spatially physical shape interpretation is limited;
- do not pretend pixels correspond to mm.

Record:
- PyRadiomics version;
- config;
- feature names.

Save:
outputs/radiomics_features.csv

==================================================
7. TARGET
==================================================

Inspect the official MMOTU label definitions.

Use the official classification task.

If a benign/malignant mapping is clearly supported by the official metadata, optionally include it.

Do NOT invent a binary mapping.

==================================================
8. SPLITTING / CV
==================================================

Use a leakage-safe 5-fold CV.

If patient IDs exist:
- group by patient;
- ensure patient independence across folds.

If no patient IDs exist:
- use stratified image-level CV if possible;
- clearly mark this as a limitation.

All data-dependent operations must happen INSIDE each training fold:
- imputation;
- scaling;
- feature selection;
- model fitting.

==================================================
9. FEATURE SELECTION
==================================================

Keep this simple.

Use one or two strategies only:
- variance filtering;
- univariate selection or L1-based selection.

Target-dependent selection must happen inside each training fold.

Track which features are selected in each fold.

Save:
outputs/selected_features_by_fold.csv

==================================================
10. MODELS
==================================================

Compare:

1. Logistic Regression
2. Random Forest
3. XGBoost

Also include a DummyClassifier baseline.

Use sensible defaults plus LIGHT tuning.

Maximum:
- about 10–20 configurations per model;
- randomized search preferred;
- CPU only.

Do NOT run massive grid searches.

==================================================
11. METRICS
==================================================

For binary classification:
- ROC-AUC
- PR-AUC
- sensitivity
- specificity
- precision
- F1
- balanced accuracy
- confusion matrix
- Brier score
- calibration curve

For multiclass:
adapt appropriately:
- macro F1
- weighted F1
- per-class recall
- confusion matrix
- multiclass AUC if valid

Report mean and standard deviation across CV folds.

==================================================
12. OVERFITTING CHECK
==================================================

For each model report:
- training performance;
- validation performance;
- train-validation gap.

Flag suspiciously large gaps.

==================================================
13. CALIBRATION
==================================================

Generate one calibration plot per applicable model.

Do not perform complex recalibration unless straightforward.

The goal is to demonstrate that calibration matters, not optimize it heavily.

==================================================
14. OUTPUT REPORT
==================================================

Generate:

outputs/demo_report.html

Audience:
clinical researcher, not ML engineer.

Keep it concise.

Sections:
1. Research question
2. Dataset
3. Data quality
4. Validation strategy
5. Radiomic features
6. Models
7. Results
8. Calibration
9. Overfitting check
10. Methodological safeguards
11. Limitations
12. What this result does NOT prove

Include:
- class distribution;
- ROC curves;
- confusion matrices;
- calibration plots;
- model comparison table.

==================================================
15. METHODOLOGICAL SAFEGUARDS
==================================================

The report must explicitly show:

- whether patient grouping was available;
- feature selection performed inside CV;
- scaling fitted only on training folds;
- no tuning on test data;
- random seed recorded;
- no fabricated clinical variables;
- no fabricated physical spacing.

==================================================
16. TESTS
==================================================

Write at least these automated tests:

1. No group leakage across folds, where grouping exists.
2. Scaler is fit only on fold-training data.
3. Feature selector is fit only on fold-training data.
4. Same random seed produces same splits.

Run all tests and fix failures before completion.

==================================================
17. README
==================================================

README must explain:

- what the demo does;
- how to install;
- how to obtain MMOTU;
- how to run:
  python run_demo.py
- what files are produced;
- methodological safeguards;
- known limitations;
- how this pipeline could later be adapted to a Gemelli/IOTA-like dataset.

==================================================
18. COMPUTE BUDGET
==================================================

Keep the whole project lightweight.

Rules:
- CPU only;
- no deep learning;
- no segmentation training;
- no large pretrained models;
- no massive tuning;
- no more than 20 configurations per model;
- prefer a subset of the dataset for quick debugging if needed;
- once code works, run the full feasible dataset.

If something becomes expensive:
prefer methodological completeness over exhaustive optimization.

==================================================
19. SELF-REVIEW
==================================================

After the demo runs successfully, perform ONE methodological review.

Check specifically:
- leakage;
- wrong split logic;
- feature-selection leakage;
- invalid calibration;
- overclaiming;
- incorrect use of patient/image independence;
- fabricated assumptions.

Fix issues found.

Do not perform multiple long review cycles.

==================================================
20. FINAL HANDOFF
==================================================

At the end, give me:

1. what was built;
2. what dataset was actually used;
3. whether patient-level grouping was available;
4. exact command to run the demo;
5. key files generated;
6. main methodological safeguards;
7. strongest limitations;
8. 5-minute demo sequence for a gynecologic oncology researcher.

Very important:
Do not claim clinical validity.
Do not optimize for impressive AUC.
Optimize for:
- reproducibility;
- correct methodology;
- clarity;
- fast reuse on a future hospital dataset.

Start now.
Inspect MMOTU first, then implement the smallest complete version that satisfies these requirements.


==================================================
21. JUPITER NOTEBOOK
==================================================

In addition to the modular Python project and HTML report, create a polished Jupyter notebook called demo_pipeline.ipynb. The notebook should be the main presentation artifact for a clinical researcher. It must narrate the complete experiment from dataset inspection to radiomics extraction, leakage-safe cross-validation, model comparison, calibration, methodological safeguards and limitations. Keep implementation code minimal by importing reusable functions from src/. The notebook must execute top-to-bottom without manual edits and should contain concise explanatory Markdown between steps.