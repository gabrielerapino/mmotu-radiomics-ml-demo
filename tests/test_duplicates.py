import numpy as np
import pandas as pd
from PIL import Image

from src.duplicates import build_duplicate_groups, clean_validation_indices
from src.pipeline import make_cv_splits


def test_decoded_pixel_duplicate_grouping_is_deterministic(tmp_path):
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    dataset = tmp_path / "OTU_2d"
    first = dataset / "images" / "first.png"
    second = tmp_path / "figshare_mmotu" / "train" / "train_image" / "first.png"
    different = dataset / "images" / "different.png"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    Image.fromarray(pixels).save(first, compress_level=1)
    Image.fromarray(pixels).save(second, compress_level=9)
    Image.fromarray(np.flipud(pixels)).save(different, compress_level=1)
    labels = pd.DataFrame({
        "filename": ["first.png", "different.png"],
        "label": [0, 1],
        "label_name": ["a", "b"],
        "official_split": ["train", "train"],
        "image_path": [first, different],
    })

    groups_a = build_duplicate_groups(labels, dataset)
    groups_b = build_duplicate_groups(labels, dataset)
    assert groups_a[['filename', 'duplicate_group_id', 'source_folder']].to_dict('records') == groups_b[
        ['filename', 'duplicate_group_id', 'source_folder']
    ].to_dict('records')
    first_copies = groups_a.loc[groups_a.filename == 'first.png']
    assert len(first_copies) == 2
    assert first_copies.duplicate_group_id.nunique() == 1
    assert first_copies.file_hash.nunique() == 2
    assert first_copies.pixel_hash.nunique() == 1
    assert first_copies.is_modeling_observation.sum() == 1
    other = groups_a.loc[groups_a.filename == 'different.png'].iloc[0]
    assert first_copies.iloc[0].duplicate_group_id != other.duplicate_group_id


def test_duplicate_grouped_cv_keeps_group_ids_within_one_fold():
    labels = np.repeat(np.arange(4), 10)
    groups = np.repeat(np.arange(20), 2)
    splits = make_cv_splits(labels, seed=42, n_splits=5, groups=groups)
    fold_for_group = {}
    for fold, (_, validation_indices) in enumerate(splits):
        for group in set(groups[validation_indices]):
            assert group not in fold_for_group
            fold_for_group[group] = fold
    assert len(fold_for_group) == len(np.unique(groups))


def test_contaminated_validation_duplicate_is_removed():
    kept = clean_validation_indices(
        ["pixel-a", "pixel-b"],
        ["pixel-c", "pixel-a", "pixel-d", "pixel-b"],
    )
    assert kept.tolist() == [0, 2]
    assert not set(["pixel-c", "pixel-d"]) & set(["pixel-a", "pixel-b"])
