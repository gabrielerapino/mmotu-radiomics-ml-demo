# MMOTU ovarian ultrasound radiomics demo

This small research demonstrator loads the public **MMOTU OTU_2d** ultrasound images, expert masks, and the dataset's eight image-category labels. It audits the data, extracts a conservative radiomic feature set, compares a dummy baseline with logistic regression, random forest, and XGBoost, and creates a researcher-facing HTML report and notebook.

This is a research demonstration. It is not a clinical diagnostic tool and does not establish clinical validity.

## Data availability

The MMOTU images, masks, and source archives are not included in this repository. Use `python download_data.py` to obtain the public dataset from its upstream source. Generated per-image radiomic features, file inventories, and individual prediction records are also kept out of the public repository; the aggregate report and aggregate comparison artifacts are included.

## Install

Python 3.10–3.12 is recommended. CPU only.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The standard install uses PyRadiomics from its pinned official source commit. The extractor computes first-order, GLCM, GLRLM, and 2D shape features; if PyRadiomics is unavailable, the explicitly labeled NumPy fallback keeps the demo runnable. The actual engine, version, settings, and feature names are recorded in the outputs.

## Obtain MMOTU

The repository authors link the public Google Drive folder from the [MMOTU_DS2Net repository](https://github.com/cv516Buaa/MMOTU_DS2Net). Download it with:

```bash
python download_data.py
```

The expected path is `data/raw/OTU_2d`, with the official `train_cls.txt` and `val_cls.txt`, an `images/` directory, and an `annotations/` directory. Google Drive can take time to enumerate the large annotation folder. The downloader skips valid image and mask files already present, so it can resume without clearing the partial download. If Drive throttles access, download the original public files by hand into the stated layout. The script will use an explicitly labeled synthetic fallback if the real dataset or complete matching files are unavailable, so inspect `outputs/data_audit.json` before interpreting any result.

## Run

```bash
python run_demo.py
```

To exercise the software without MMOTU:

```bash
python run_demo.py --synthetic
```

Run the presentation notebook with Jupyter installed:

```bash
jupyter notebook demo_pipeline.ipynb
```

## Outputs

- `outputs/data_audit.json`: counts, class balance, file integrity, dimensions, metadata, and patient-ID availability.
- `outputs/duplicate_groups.csv`: one row per physical source copy of every usable labeled image, with byte and decoded-pixel SHA256 hashes, duplicate-group ID, and a flag identifying the copy selected by the loader for modeling.
- `outputs/duplicate_leakage_audit.json`: duplicate-cluster counts, original fold and official partition crossings, and validation cleaning counts.
- `outputs/radiomics_features.csv`: image labels and extracted feature rows.
- `outputs/radiomics_config.json`: extraction engine, feature names, version, and spacing note.
- `outputs/selected_features_by_fold.csv`: selected features for each model and duplicate-grouped CV fold.
- `outputs/selected_features_image_level_original.csv`: selected features in the original image-level CV.
- `outputs/selected_features_duplicate_safe_by_fold.csv`: features selected under duplicate-grouped CV.
- `outputs/duplicate_safe_model_comparison.csv`: original CV versus duplicate-safe CV and original versus cleaned validation metrics.
- `outputs/duplicate_safe_validation.json`: both original official-validation and duplicate-clean validation results, with class distributions.
- `outputs/metrics.json`: fold and official-validation metrics, probabilities, and confusion matrices.
- `outputs/demo_report.html`: concise report with model comparison, confusion matrices, and calibration curves.
- `outputs/figures/`: saved report figures.

The files above are generated locally by `python run_demo.py`; the repository tracks the HTML report, aggregate duplicate audit and model comparison, radiomics configuration, fold-level feature selections, and figures. Per-image and detailed prediction outputs remain local.

Open `outputs/demo_report.html` in a browser and select **Export PDF** to open the print dialog; choose **Save as PDF** to save a PDF that follows the report's print layout. The export uses the browser's native HTML renderer, so no PDF package or separate conversion step is required.

## Methods and safeguards

The official 8-class category task is used; no benign/malignant remapping is imposed. Five-fold duplicate-grouped image-level CV runs on the official training partition, with the original stratified image-level CV retained as a comparator. The official validation partition is held out; rows whose decoded pixels also occur in official training are excluded from the duplicate-clean validation results, while the original validation results are retained. Imputation, variance filtering, scaling, and mutual-information feature selection are fitted inside each training fold. The seed is recorded. The data lists identify images by filename but do not provide reliable patient identifiers, so duplicate isolation does not guarantee patient independence. The pipeline does not create clinical covariates or physical pixel spacing. Shape measurements are in pixel units.

## Limitations

Duplicate-grouped image-level folds may still include different images from the same patient in both training and validation. Performance from one public source may not transfer to another hospital or acquisition workflow. The dataset is small and imbalanced, and the masks/labels encode annotation decisions. No result from this demo proves clinical validity, diagnostic utility, or patient-level performance.

## Adapting to a Gemelli/IOTA-like dataset

Keep the feature extraction and fold-local modeling code, then add a new loader that maps local image/mask files to an approved, de-identified study table. Use a stable patient key only if the data governance process permits it; pass that key to a group-aware splitter and verify that no patient crosses folds. Define outcomes from the study protocol and reference standard, preserve missingness and acquisition metadata, and record physical spacing only when it is supplied by the source. Validate the extraction settings and label semantics with the clinical research team before running the same comparison. The demo contains no Gemelli or IOTA patient data.

## Five-minute demo sequence

1. Open the notebook and read the research question and dataset provenance (about 45 seconds).
2. Review the audit table and patient-grouping limitation (about 60 seconds).
3. Show how masks define the radiomic region and review the feature families (about 60 seconds).
4. Compare the CV table and calibration/confusion plots (about 90 seconds).
5. Close with safeguards, limits, and what the experiment does not prove (about 45 seconds).

## Tests

```bash
pytest -q
```
