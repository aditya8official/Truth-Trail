"""Audit a saved split manifest for exact overlap and filename-based clues.

Filename groups are hypotheses only. Supply a documented source-group file
before treating related records as the same source or rebuilding a split.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def fake_numeric_group(path: str) -> str | None:
    """Return the numeric token from names like easy_156_0010.jpg."""
    stem = Path(path).stem
    pieces = stem.split("_")
    if len(pieces) >= 3 and pieces[0] in {"easy", "mid", "hard"} and pieces[1].isdigit():
        return pieces[1]
    return None


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(manifest_path: Path, dataset_root: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    split_names = [name for name, value in manifest.items() if isinstance(value, list)]
    records = {
        split: manifest[split]
        for split in split_names
    }

    path_sets = {
        split: {record["path"] for record in split_records}
        for split, split_records in records.items()
    }
    path_overlaps = {}
    basename_splits: dict[str, set[str]] = defaultdict(set)
    numeric_group_splits: dict[str, set[str]] = defaultdict(set)
    difficulty_numeric_group_splits: dict[str, set[str]] = defaultdict(set)
    path_rows: list[str] = []
    folder_label_mismatches = []

    for split, split_records in records.items():
        for record in split_records:
            source_path = record["path"]
            path = Path(source_path)
            path_rows.append(source_path)
            basename_splits[path.name].add(split)
            group = fake_numeric_group(source_path)
            if group is not None:
                numeric_group_splits[group].add(split)
                stem_parts = path.stem.split("_")
                difficulty_numeric_group_splits[
                    "_".join(stem_parts[:2])
                ].add(split)

            expected_folder = "training_real" if record["label"] == 0 else "training_fake"
            if f"/{expected_folder}/" not in source_path.replace("\\", "/"):
                folder_label_mismatches.append({"split": split, "path": source_path})

    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            path_overlaps[f"{left}:{right}"] = len(path_sets[left] & path_sets[right])

    basename_crossings = {
        name: sorted(split_set)
        for name, split_set in basename_splits.items()
        if len(split_set) > 1
    }
    numeric_group_crossings = {
        group: sorted(split_set)
        for group, split_set in numeric_group_splits.items()
        if len(split_set) > 1
    }
    difficulty_numeric_crossings = {
        group: sorted(split_set)
        for group, split_set in difficulty_numeric_group_splits.items()
        if len(split_set) > 1
    }

    content_hashes: dict[str, set[str]] = defaultdict(set)
    missing_images = 0
    if dataset_root is not None:
        for split, split_records in records.items():
            for record in split_records:
                filename = Path(record["path"]).name
                folder = "training_real" if record["label"] == 0 else "training_fake"
                image_path = dataset_root / folder / filename
                if not image_path.is_file():
                    missing_images += 1
                    continue
                content_hashes[digest_file(image_path)].add(split)

    content_hash_crossings = sum(
        1 for split_set in content_hashes.values() if len(split_set) > 1
    )

    return {
        "seed": manifest.get("seed"),
        "split_counts": {
            split: {
                "total": len(split_records),
                "real": sum(record["label"] == 0 for record in split_records),
                "fake": sum(record["label"] == 1 for record in split_records),
            }
            for split, split_records in records.items()
        },
        "exact_path_overlaps": path_overlaps,
        "duplicate_path_rows_within_manifest": sum(
            count - 1 for count in Counter(path_rows).values() if count > 1
        ),
        "duplicate_basenames_across_splits": basename_crossings,
        "fake_numeric_filename_groups": len(numeric_group_splits),
        "fake_numeric_groups_crossing_split_count": len(numeric_group_crossings),
        "fake_numeric_groups_crossing_split_examples": dict(
            list(sorted(numeric_group_crossings.items()))[:12]
        ),
        "fake_difficulty_numeric_groups_crossing_splits": difficulty_numeric_crossings,
        "label_folder_mismatches": folder_label_mismatches,
        "content_hash_groups_crossing_splits": content_hash_crossings if dataset_root else None,
        "missing_image_files": missing_images if dataset_root else None,
        "note": (
            "The numeric token in fake filenames is not documented here as a source identity. "
            "Cross-split numeric groups are review candidates, not evidence of leakage."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to dataset_split_seed42.json")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional directory containing training_real/ and training_fake/",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, args.dataset_root), indent=2))


if __name__ == "__main__":
    main()
