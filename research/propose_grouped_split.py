"""Propose a candidate group-aware split from a saved split manifest.

This is a sensitivity-analysis tool. It assumes that a fake filename's middle
numeric token identifies a source group and that a matching real filename
(``real_NNNNN``) belongs to that same group. The dataset documentation does
not confirm this assumption. The existing split is never modified.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys


FAKE_NAME = re.compile(r"^(?:easy|mid|hard)_(\d+)_\d+$")
REAL_NAME = re.compile(r"^real_(\d+)$")


def candidate_group(path: str) -> str:
    """Return the provisional group key inferred from a manifest path."""
    stem = Path(path).stem
    real_match = REAL_NAME.fullmatch(stem)
    if real_match:
        return f"candidate-id:{int(real_match.group(1))}"

    fake_match = FAKE_NAME.fullmatch(stem)
    if fake_match:
        return f"candidate-id:{int(fake_match.group(1))}"

    # Keep unrecognized records intact as individual rows; do not silently
    # infer a relationship from an unfamiliar filename convention.
    return f"unparsed:{path}"


def read_manifest(path: Path) -> tuple[dict, list[str], dict[str, list[dict]]]:
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object.")

    split_names = [name for name, value in manifest.items() if isinstance(value, list)]
    if len(split_names) < 2:
        raise ValueError("Manifest must contain at least two split arrays.")

    rows = {split: manifest[split] for split in split_names}
    seen_paths: set[str] = set()
    for split, records in rows.items():
        for record in records:
            if not isinstance(record, dict) or "path" not in record or "label" not in record:
                raise ValueError(f"Invalid record in split {split!r}; expected path and label.")
            if record["label"] not in (0, 1):
                raise ValueError(f"Unexpected label in split {split!r}: {record['label']!r}")
            if record["path"] in seen_paths:
                raise ValueError(f"Duplicate path in manifest: {record['path']}")
            seen_paths.add(record["path"])
    if not seen_paths:
        raise ValueError("Manifest has no records.")
    return manifest, split_names, rows


def summarize(rows: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "total": len(records),
            "real": sum(record["label"] == 0 for record in records),
            "fake": sum(record["label"] == 1 for record in records),
        }
        for split, records in rows.items()
    }


def propose(
    manifest: dict,
    split_names: list[str],
    rows: dict[str, list[dict]],
    seed: int,
) -> tuple[dict[str, list[dict]], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    original_order: dict[str, int] = {}
    for split in split_names:
        for record in rows[split]:
            groups[candidate_group(record["path"])].append(record)
            original_order[record["path"]] = len(original_order)

    original_targets = summarize(rows)
    total_by_label = {
        label: sum(counts[name] for counts in original_targets.values())
        for label, name in ((0, "real"), (1, "fake"))
    }
    total_rows = sum(total_by_label.values())

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    proposed = {split: [] for split in split_names}
    counts = {split: {"total": 0, "real": 0, "fake": 0} for split in split_names}
    processed = 0

    for _, group_records in group_items:
        group_counts = {
            "total": len(group_records),
            "real": sum(record["label"] == 0 for record in group_records),
            "fake": sum(record["label"] == 1 for record in group_records),
        }
        projected_fraction = (processed + group_counts["total"]) / total_rows

        def placement_score(destination: str) -> float:
            score = 0.0
            for split in split_names:
                for label_name, label in (("real", 0), ("fake", 1)):
                    current = counts[split][label_name]
                    if split == destination:
                        current += group_counts[label_name]
                    split_fraction = (
                        original_targets[split][label_name]
                        / max(total_by_label[label], 1)
                    )
                    ideal = total_by_label[label] * split_fraction * projected_fraction
                    score += ((current - ideal) / max(original_targets[split][label_name], 1)) ** 2
            return score

        destination = min(split_names, key=placement_score)
        proposed[destination].extend(group_records)
        for key in ("total", "real", "fake"):
            counts[destination][key] += group_counts[key]
        processed += group_counts["total"]

    for split in split_names:
        proposed[split].sort(key=lambda record: original_order[record["path"]])

    # By construction, each inferred group is assigned as one indivisible unit.
    proposed_split_by_path = {
        record["path"]: split
        for split, records in proposed.items()
        for record in records
    }
    crossing_groups = sum(
        len({proposed_split_by_path[record["path"]] for record in group_records}) > 1
        for group_records in groups.values()
    )
    if crossing_groups:
        raise RuntimeError("Internal error: a candidate group spans proposed splits.")

    report = {
        "assumption": (
            "The numeric token in a fake filename and the numeric ID in a real filename "
            "identify the same source group. This is unconfirmed."
        ),
        "seed": seed,
        "candidate_group_count": len(groups),
        "unparsed_singleton_row_count": sum(
            1 for records in groups.values() if len(records) == 1
            and candidate_group(records[0]["path"]).startswith("unparsed:")
        ),
        "largest_candidate_group_size": max(map(len, groups.values())),
        "candidate_groups_crossing_proposed_splits": crossing_groups,
        "original_split_counts": original_targets,
        "proposed_split_counts": summarize(proposed),
        "manifest_written": False,
    }
    return proposed, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to the existing split manifest")
    parser.add_argument("--seed", type=int, help="Assignment seed (defaults to manifest seed or 42)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a separate candidate manifest; never overwrites an existing file",
    )
    parser.add_argument(
        "--accept-filename-link-assumption",
        action="store_true",
        help="Required with --output to acknowledge the unconfirmed filename grouping assumption",
    )
    args = parser.parse_args()

    if args.output and not args.accept_filename_link_assumption:
        parser.error("--output requires --accept-filename-link-assumption")
    if args.output and args.output.resolve() == args.manifest.resolve():
        parser.error("--output must be separate from the existing manifest")
    if args.output and args.output.exists():
        parser.error(f"Output already exists; refusing to overwrite: {args.output}")

    try:
        manifest, split_names, rows = read_manifest(args.manifest)
        seed = args.seed if args.seed is not None else manifest.get("seed", 42)
        proposed, report = propose(manifest, split_names, rows, seed)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    if args.output:
        source_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        candidate_manifest = {
            "seed": seed,
            "group_split": {
                "method": "deterministic greedy class-balanced candidate grouping",
                "assumption": report["assumption"],
                "source_manifest_sha256": source_hash,
                "source_manifest_name": args.manifest.name,
                "warning": "Sensitivity-analysis candidate only; grouping semantics are unconfirmed.",
            },
            **proposed,
        }
        try:
            with args.output.open("x") as output_file:
                json.dump(candidate_manifest, output_file, indent=2)
                output_file.write("\n")
        except FileExistsError:
            parser.error(f"Output appeared during execution; refusing to overwrite: {args.output}")
        report["manifest_written"] = True
        report["output"] = str(args.output)

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
