"""Conservative PyRadiomics extraction with a transparent lightweight fallback."""
from __future__ import annotations
import json
import logging
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def _load_array(value, mode="L"):
    if isinstance(value, np.ndarray):
        return value
    with Image.open(value) as im:
        return np.asarray(im.convert(mode))


@lru_cache(maxsize=1)
def _get_pyradiomics_extractor():
    try:
        from radiomics import featureextractor
        import SimpleITK as sitk
    except ImportError:
        return None
    class _GlcmSymmetryNotice(logging.Filter):
        def filter(self, record):
            return "GLCM is symmetrical, therefore Sum Average = 2 * Joint Average" not in record.getMessage()

    glcm_logger = logging.getLogger("radiomics.glcm")
    if not any(isinstance(f, _GlcmSymmetryNotice) for f in glcm_logger.filters):
        glcm_logger.addFilter(_GlcmSymmetryNotice())
    extractor = featureextractor.RadiomicsFeatureExtractor(
        label=1, binWidth=16, normalize=False, removeOutliers=None
    )
    extractor.disableAllFeatures()
    for feature_class in ("firstorder", "glcm", "glrlm", "shape2D"):
        extractor.enableFeatureClassByName(feature_class)
    return extractor, sitk


def _scalar_features(result) -> dict:
    features = {}
    for name, value in result.items():
        if not str(name).startswith("original_"):
            continue
        scalar = np.asarray(value)
        if scalar.ndim != 0:
            continue
        value = scalar.item()
        if isinstance(value, complex):
            if not np.isclose(value.imag, 0.0, atol=1e-12):
                raise ValueError(f"PyRadiomics feature {name} has a non-real value: {value}")
            value = value.real
        features[str(name)] = float(value)
    return features


def extract_one(image, mask, label=1) -> dict:
    image_array = _load_array(image).astype(np.float32)
    mask_array = _load_array(mask) > 0
    if image_array.shape != mask_array.shape:
        raise ValueError(f"Image and mask dimensions differ: {image_array.shape} vs {mask_array.shape}")
    if not mask_array.any():
        raise ValueError("Cannot extract radiomics from an empty mask")
    pyradiomics = _get_pyradiomics_extractor()
    if pyradiomics is None:
        return _fallback_features(image_array, mask_array)
    extractor, sitk = pyradiomics
    if int(label) != 1:
        extractor.settings["label"] = int(label)
    img = sitk.GetImageFromArray(image_array)
    msk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    return _scalar_features(extractor.execute(img, msk))


def _fallback_features(image, mask):
    values = image[mask]
    feats = {f"fallback_firstorder_{name}": float(func(values)) for name, func in {
        "mean": np.mean, "std": np.std, "median": np.median, "min": np.min,
        "max": np.max, "p10": lambda x: np.percentile(x, 10),
        "p90": lambda x: np.percentile(x, 90),
    }.items()}
    feats["fallback_shape_area_pixels"] = float(mask.sum())
    feats["fallback_shape_extent"] = float(mask.mean())
    # Basic gray-level co-occurrence summaries at a one-pixel horizontal offset.
    q = np.clip((image / 16).astype(int), 0, 15)
    valid = mask[:, :-1] & mask[:, 1:]
    a, b = q[:, :-1][valid], q[:, 1:][valid]
    if len(a):
        p = np.zeros((16, 16), dtype=float)
        np.add.at(p, (a, b), 1); p /= p.sum()
        ii, jj = np.indices(p.shape)
        feats.update({"fallback_glcm_contrast": float(((ii-jj)**2*p).sum()),
                      "fallback_glcm_homogeneity": float((p/(1+np.abs(ii-jj))).sum()),
                      "fallback_glcm_energy": float((p*p).sum())})
    else:
        feats.update({"fallback_glcm_contrast": 0.0, "fallback_glcm_homogeneity": 0.0,
                      "fallback_glcm_energy": 0.0})
    return feats


def extract_features(frame: pd.DataFrame, images=None, masks=None) -> tuple[pd.DataFrame, dict]:
    rows, names = [], set()
    for index, row in frame.iterrows():
        image = images[index] if images is not None else row.image_path
        mask = masks[index] if masks is not None else row.mask_path
        feature = extract_one(image, mask)
        names.update(feature); rows.append(feature)
    out = pd.DataFrame(rows).reindex(columns=sorted(names)).replace([np.inf, -np.inf], np.nan)
    try:
        import radiomics
        version, engine = radiomics.__version__, "PyRadiomics firstorder + GLCM + GLRLM + 2D shape"
        settings = {"2d": True, "feature_classes": ["firstorder", "glcm", "glrlm", "shape2D"],
                    "bin_width": 16, "normalize": False,
                    "spacing": "pixel units; physical spacing unavailable"}
    except ImportError:
        version, engine = None, "transparent NumPy fallback: first-order, simple GLCM, pixel geometry"
        settings = {"2d": True, "spacing": "pixel units; physical spacing unavailable"}
    return out, {"engine": engine, "pyradiomics_version": version,
                 "feature_names": list(out.columns), "settings": settings}


def save_features(frame: pd.DataFrame, labels: pd.DataFrame, path: Path) -> None:
    combined = pd.concat([labels[["filename", "label", "label_name", "official_split"]].reset_index(drop=True),
                          frame.reset_index(drop=True)], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
