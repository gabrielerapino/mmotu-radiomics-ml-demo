"""Exact duplicate grouping and duplicate-safe validation helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_pixel_hash(pixels: np.ndarray) -> str:
    pixels = np.ascontiguousarray(pixels)
    digest = hashlib.sha256()
    digest.update(str(pixels.dtype).encode("ascii"))
    digest.update(json.dumps(pixels.shape, separators=(",", ":")).encode("ascii"))
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def decoded_pixel_hash(path: Path) -> str:
    """Hash the grayscale pixel array used by the radiomics loader.

    Shape and dtype are included in the digest so reshaped arrays cannot collide.
    This identifies exact decoded-pixel equality even when encoded files differ.
    """
    with Image.open(path) as image:
        pixels = np.ascontiguousarray(np.asarray(image.convert("L")))
    return _array_pixel_hash(pixels)


def build_duplicate_groups(labels: pd.DataFrame, data_root: Path, images=None) -> pd.DataFrame:
    """Return deterministic records for each source copy of every usable image.

    `is_modeling_observation` identifies the single file path selected by the
    current loader. Alternate source copies are audited but are not extra rows
    in the model matrices.
    """
    data_root = Path(data_root)
    common_root = data_root.parent.resolve()
    image_roots = [data_root / "images" if (data_root / "images").is_dir() else data_root]
    archive_root = data_root.parent / "figshare_mmotu"
    if archive_root.is_dir():
        image_roots.append(archive_root)
    paths_by_stem: dict[str, list[Path]] = {}
    for image_root in image_roots:
        for path in image_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            relative_parts = {part.lower() for part in path.relative_to(image_root).parts[:-1]}
            if image_root == archive_root and not relative_parts & {"image", "train_image"}:
                continue
            paths_by_stem.setdefault(path.stem.lower(), []).append(path)
    rows = []
    for index, row in enumerate(labels.itertuples(index=False)):
        raw_path = getattr(row, "image_path", None)
        if raw_path is not None and not pd.isna(raw_path):
            selected_path = Path(raw_path)
            source_paths = paths_by_stem.get(Path(row.filename).stem.lower(), [])
            if selected_path not in source_paths:
                source_paths.append(selected_path)
            source_paths = sorted(set(source_paths), key=lambda path: str(path))
            for image_path in source_paths:
                file_hash = sha256_file(image_path)
                pixel_hash = decoded_pixel_hash(image_path)
                try:
                    source_folder = str(image_path.resolve().parent.relative_to(common_root))
                except ValueError:
                    source_folder = os.path.relpath(image_path.resolve().parent, common_root)
                rows.append({
                    "duplicate_group_id": f"pixel_{pixel_hash}",
                    "filename": str(row.filename),
                    "label": int(row.label),
                    "label_name": str(getattr(row, "label_name", row.label)),
                    "source_partition": str(getattr(row, "official_split", "unknown")),
                    "source_folder": source_folder,
                    "file_hash": file_hash,
                    "pixel_hash": pixel_hash,
                    "is_modeling_observation": image_path.resolve() == selected_path.resolve(),
                })
        elif images is not None:
            file_hash = ""
            pixel_hash = _array_pixel_hash(np.asarray(images[index], dtype=np.uint8))
            source_folder = "synthetic in-memory array"
            rows.append({
                "duplicate_group_id": f"pixel_{pixel_hash}",
                "filename": str(row.filename),
                "label": int(row.label),
                "label_name": str(getattr(row, "label_name", row.label)),
                "source_partition": str(getattr(row, "official_split", "unknown")),
                "source_folder": source_folder,
                "file_hash": file_hash,
                "pixel_hash": pixel_hash,
                "is_modeling_observation": True,
            })
        else:
            raise ValueError(f"No readable image path for usable observation {row.filename}")
    return pd.DataFrame(rows)


def clean_validation_indices(train_group_ids, validation_group_ids) -> np.ndarray:
    """Indices of validation samples whose pixel group is absent from training."""
    training_groups = set(pd.Series(train_group_ids).astype(str))
    validation = pd.Series(validation_group_ids).astype(str)
    return np.flatnonzero(~validation.isin(training_groups).to_numpy())


def duplicate_group_sizes(groups: pd.DataFrame) -> pd.Series:
    """Counts by pixel group, sorted by the stable group ID."""
    return groups.groupby("duplicate_group_id", sort=True).size()
