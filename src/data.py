"""MMOTU loading, validation, and an explicitly synthetic fallback."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = {
    0: "chocolate cyst", 1: "serous cystadenoma", 2: "teratoma",
    3: "theca cell tumor", 4: "simple cyst", 5: "normal ovary",
    6: "mucinous cystadenoma", 7: "high-grade serous carcinoma",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_label_file(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError(f"Unexpected label row in {path}: {line!r}")
        rows.append((parts[0], int(parts[1])))
    return pd.DataFrame(rows, columns=["filename", "label"])


def _find_label_file(root: Path, stem: str) -> Path | None:
    candidates = [root / f"{stem}_cls.txt", *root.glob(f"**/{stem}_cls.txt")]
    return next((p for p in candidates if p.is_file()), None)


def _image_map(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            result.setdefault(p.name.lower(), p)
    return result


def _figshare_image_map(root: Path) -> dict[str, Path]:
    """Index OTU_2D images extracted from the public Figshare archive."""
    result: dict[str, Path] = {}
    image_dirs = {"image", "train_image"}
    for p in root.rglob("*"):
        if (p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                and any(part.lower() in image_dirs for part in p.parts)):
            result.setdefault(p.name.lower(), p)
    return result


def _mask_map(root: Path) -> dict[str, Path]:
    """Index one binary annotation variant by the source image stem."""
    result: dict[str, Path] = {}
    files = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=lambda p: (0 if p.stem.lower().endswith("_binary") else 1, str(p)))
    for p in files:
        stem = p.stem.lower()
        if stem.endswith("_binary_binary"):
            continue
        if stem.endswith("_binary"):
            stem = stem[:-7]
        result.setdefault(stem, p)
    return result


def load_mmotu(root: Path) -> tuple[pd.DataFrame, dict]:
    """Load official train/validation labels and match image and mask files by filename."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MMOTU directory not found: {root}")
    train_file, val_file = _find_label_file(root, "train"), _find_label_file(root, "val")
    if train_file is None or val_file is None:
        raise FileNotFoundError("Expected official train_cls.txt and val_cls.txt in OTU_2d")
    labels = pd.concat([
        read_label_file(train_file).assign(official_split="train"),
        read_label_file(val_file).assign(official_split="validation"),
    ], ignore_index=True)
    labels["label_name"] = labels.label.map(CLASS_NAMES)
    unknown_labels = sorted(set(labels.label)-set(CLASS_NAMES))
    if unknown_labels:
        raise ValueError(f"Official class list contains undocumented class IDs: {unknown_labels}")
    image_folder = root / "images"
    images = _image_map(image_folder if image_folder.exists() else root)
    archive_root = root.parent / "figshare_mmotu"
    if archive_root.is_dir():
        for name, path in _figshare_image_map(archive_root).items():
            images.setdefault(name, path)
    masks = {}
    for name in ["annotations", "masks", "labels"]:
        folder = root / name
        if folder.exists():
            masks.update(_mask_map(folder))
    # Some copies store the annotation folder with different casing.
    for folder in root.iterdir():
        if folder.is_dir() and any(k in folder.name.lower() for k in ("annot", "mask")):
            masks.update(_mask_map(folder))
    if archive_root.is_dir():
        archive_masks = {}
        for p in archive_root.rglob("*"):
            if (p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                    and any("label" in part.lower() for part in p.parts)):
                stem = p.stem.lower()
                if stem.endswith("_binary_binary"):
                    continue
                if stem.endswith("_binary"):
                    stem = stem[:-7]
                archive_masks.setdefault(stem, p)
        for stem, path in archive_masks.items():
            masks.setdefault(stem, path)
    images_by_stem = {Path(k).stem.lower(): v for k,v in images.items()}
    labels["image_path"] = labels.filename.map(lambda x: images_by_stem.get(Path(x).stem.lower()))
    labels["mask_path"] = labels.filename.map(lambda x: masks.get(Path(x).stem.lower()))
    labels["patient_id"] = None  # The public label lists contain image IDs only.
    audit = audit_dataset(labels, images, masks, root)
    duplicate_rows = labels.filename.duplicated(keep="first")
    invalid = set(audit["missing_images_for_labeled_rows"] + audit["missing_masks_for_labeled_rows"] +
                  audit["empty_masks"] + [x["filename"] for x in audit["image_mask_mismatches"]])
    bad_rows = labels.filename.isin(invalid) | duplicate_rows
    usable = labels.loc[~bad_rows].copy().reset_index(drop=True)
    audit["excluded_rows_from_modeling"] = len(labels)-len(usable)
    audit["usable_labeled_rows"] = len(usable)
    return usable, audit


def audit_dataset(labels: pd.DataFrame, images: dict, masks: dict, root: Path) -> dict:
    image_shapes, mask_shapes, empty_masks, mismatches = [], [], [], []
    missing_image, missing_mask = [], []
    for row in labels.itertuples():
        image_path, mask_path = row.image_path, row.mask_path
        if not isinstance(image_path, Path):
            missing_image.append(row.filename)
            continue
        try:
            with Image.open(image_path) as im:
                image_shape = [im.height, im.width]
        except Exception as exc:
            mismatches.append({"filename": row.filename, "issue": f"image unreadable: {exc}"})
            continue
        image_shapes.append(tuple(image_shape))
        if not isinstance(mask_path, Path):
            missing_mask.append(row.filename)
            continue
        try:
            with Image.open(mask_path) as im:
                mask = np.asarray(im)
            mask_shape = list(mask.shape[:2])
            mask_shapes.append(tuple(mask_shape))
            if not np.any(mask):
                empty_masks.append(row.filename)
            if mask_shape != image_shape:
                mismatches.append({"filename": row.filename, "image_shape": image_shape, "mask_shape": mask_shape})
        except Exception as exc:
            mismatches.append({"filename": row.filename, "issue": f"mask unreadable: {exc}"})
    duplicate_names = labels.filename[labels.filename.duplicated(keep=False)].unique().tolist()
    image_files=[]
    image_roots = [root]
    archive_root = root.parent / "figshare_mmotu"
    if archive_root.is_dir():
        image_roots.append(archive_root)
    for image_root in image_roots:
        for p in image_root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_parts = [part.lower() for part in p.relative_to(image_root).parts[:-1]]
            if any(part in {"annotations","masks","labels","label","train_label","test"}
                   for part in relative_parts) and image_root == root:
                continue
            if image_root == archive_root and not any(part in {"image", "train_image"} for part in relative_parts):
                continue
            image_files.append(p)
    image_groups = {}
    for path in image_files:
        image_groups.setdefault(path.name.lower(), []).append(path)
    duplicate_images, identical_image_copies = [], []
    for name, paths in image_groups.items():
        if len(paths) < 2:
            continue
        hashes = {hashlib.md5(path.read_bytes()).digest() for path in paths}
        (duplicate_images if len(hashes) > 1 else identical_image_copies).append(name)
    classes = {CLASS_NAMES.get(int(k), str(k)): int(v) for k, v in labels.label.value_counts().sort_index().items()}
    return {
        "dataset": "MMOTU OTU_2d", "root": str(root), "n_label_rows": int(len(labels)),
        "n_images_found": int(len(images)), "n_masks_found": int(len(masks)),
        "missing_images_for_labeled_rows": missing_image, "missing_masks_for_labeled_rows": missing_mask,
        "duplicate_filenames_in_label_lists": duplicate_names,
        "duplicate_image_filenames_on_disk": sorted(duplicate_images),
        "identical_duplicate_image_filenames": sorted(identical_image_copies),
        "class_distribution": classes,
        "image_dimensions_unique": [list(x) for x in sorted(set(image_shapes))],
        "mask_dimensions_unique": [list(x) for x in sorted(set(mask_shapes))],
        "empty_masks": empty_masks, "image_mask_mismatches": mismatches,
        "metadata_files": [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".tsv", ".txt"}],
        "patient_grouping_available": False,
        "patient_grouping_note": "The official class lists identify images by filename; no reliable patient identifier was supplied.",
    }


def make_synthetic_dataset(samples_per_class: int = 12, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Create artificial arrays for smoke/demo execution, never presented as clinical data."""
    rng = np.random.default_rng(seed)
    records, images, masks = [], [], []
    yy, xx = np.mgrid[-1:1:64j, -1:1:64j]
    for label in range(8):
        for i in range(samples_per_class):
            rx, ry = 0.22 + label * 0.01, 0.18 + (i % 4) * 0.01
            mask = (((xx / rx) ** 2 + (yy / ry) ** 2) < 1).astype(np.uint8)
            texture = rng.normal(0, 0.07, xx.shape)
            image = np.clip(0.5 + 0.12 * np.sin((label + 1) * xx * 5) + texture, 0, 1)
            image = np.where(mask > 0, np.clip(image + 0.18, 0, 1), image)
            images.append((image * 255).astype(np.uint8)); masks.append(mask)
            records.append({"filename": f"synthetic_{label}_{i:03d}", "label": label,
                            "label_name": f"synthetic class {label}", "official_split": "synthetic",
                            "patient_id": None, "image_path": None, "mask_path": None})
    audit = {"dataset": "synthetic fallback (artificial data)", "n_label_rows": len(records),
             "n_images_found": len(records), "n_masks_found": len(records),
             "missing_images_for_labeled_rows": [], "missing_masks_for_labeled_rows": [],
             "duplicate_filenames_in_label_lists": [],
             "class_distribution": {f"synthetic class {k}": samples_per_class for k in range(8)},
             "image_dimensions_unique": [[64, 64]], "mask_dimensions_unique": [[64, 64]],
             "empty_masks": [], "image_mask_mismatches": [], "metadata_files": [],
             "patient_grouping_available": False,
             "patient_grouping_note": "Synthetic observations do not represent patients.",
             "synthetic_data": True}
    return pd.DataFrame(records), audit, images, masks
