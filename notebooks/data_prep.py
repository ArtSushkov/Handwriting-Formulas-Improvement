"""data_prep.py — Dataset preparation for LaTeX OCR project."""

from hashlib import md5
from datasets import load_dataset, concatenate_datasets


def load_latex_ocr():
    """Load LaTeX_OCR (human_handwrite) and remove pixel-level duplicates.

    Within-train duplicates: keep first occurrence. Cross-split leakage
    (image appears in both train and val/test): remove from train (Lee et al., 2022).

    Returns:
        DatasetDict with 'train', 'validation', 'test' splits.
    """
    dataset = load_dataset("linxy/LaTeX_OCR", name="human_handwrite")

    hash_map = {}
    for split_name in dataset.keys():
        for idx, ex in enumerate(dataset[split_name]):
            h = md5(ex["image"].tobytes()).hexdigest()
            hash_map.setdefault(h, []).append((split_name, idx))

    dup_hashes = {h for h, locs in hash_map.items() if len(locs) > 1}
    n_dup_groups = len(dup_hashes)
    n_dup_images = sum(len(v) for v in [hash_map[h] for h in dup_hashes])
    print(f"Pixel-level duplicates: {n_dup_groups} groups, {n_dup_images} images")

    test_val_hashes = {h for h, locs in hash_map.items()
                       if any(s in ("test", "validation") for s, _ in locs)}

    seen_hashes = set()
    keep_indices = []
    removed_cross = removed_within = 0

    for idx, ex in enumerate(dataset["train"]):
        h = md5(ex["image"].tobytes()).hexdigest()
        if h in seen_hashes:
            removed_within += 1
            continue
        if h in test_val_hashes:
            removed_cross += 1
            continue
        seen_hashes.add(h)
        keep_indices.append(idx)

    original_size = len(dataset["train"])
    dataset["train"] = dataset["train"].select(keep_indices)
    print(f"Training set: {original_size} -> {len(dataset['train'])} "
          f"({removed_within} within-dup + {removed_cross} cross-leak removed)")

    return dataset


def load_mathwriting_subsample(subsample_size=5000, random_state=42):
    """Load a random subsample of MathWriting-human training set.

    Args:
        subsample_size: number of examples (default 5000).
        random_state: shuffle seed (default 42).

    Returns:
        Dataset with 'image' and 'text' columns.
    """
    dataset_mw = load_dataset("deepcopy/MathWriting-human")
    sub = dataset_mw["train"].shuffle(seed=random_state).select(
        range(subsample_size)
    )
    sub = sub.rename_column("latex", "text").select_columns(["image", "text"])
    print(f"MathWriting subsample: {len(sub):,} examples, "
          f"columns: {sub.column_names}")
    return sub


def get_combined_train(ocr_dataset, mw_subsample):
    """Concatenate LaTeX_OCR train with MathWriting subsample."""
    combined = concatenate_datasets([ocr_dataset["train"], mw_subsample])
    print(f"Combined training set: {len(combined):,} examples")
    return combined
