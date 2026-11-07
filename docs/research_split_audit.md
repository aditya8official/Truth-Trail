# Research split audit

## Inputs inspected

- `truthtrail_data/dataset_split_seed42.json`
- Saved outputs in `archive_01/mltruthtrail-2.ipynb`
- Model test-prediction CSVs in `truthtrail_data/results/`

These research artifacts live outside the `truthtrails/` application folder.

The manifest reports seed 42 and these class counts:

| Split | Total | Real (0) | Fake (1) |
|---|---:|---:|---:|
| Train | 1,428 | 756 | 672 |
| Validation | 306 | 162 | 144 |
| Test | 307 | 163 | 144 |

## Filename and path checks

Read-only checks on the manifest found:

- No exact path is repeated across train, validation, and test.
- No basename is repeated across splits.
- No duplicate path rows occur in the manifest.
- Every row's `training_real`/`training_fake` path agrees with its stored label.
- There are 480 fake-image numeric filename tokens. 153 tokens occur in more
  than one split when grouped across `easy`, `mid`, and `hard` prefixes.
- Grouping by both difficulty prefix and numeric token yields no multi-file
  groups crossing splits.

The numeric-token grouping is only a review heuristic. The
[source dataset README](https://github.com/minostauros/Real-and-Fake-Face-Detection)
says fake filename patterns encode which face regions were replaced, but the
local artifacts do not establish that the numeric token identifies a person,
source image, or manipulation lineage. Cross-split numeric groups therefore
do not prove leakage. The source also describes easy/mid/hard as subjective
categories rather than explicit labels for a separate task.

The saved notebook output reports no exact byte-identical duplicates in the
full image collection. That is a recorded notebook result; the image files are
not available locally to rerun its hash check. The manifest paths point into a
Colab `/content/dataset/...` directory, and no dataset images or archive were
found in the inspected local research folder.

## What this audit does not establish

- Whether images share a person, source photo, or editing lineage.
- Whether generated fake images are paired with particular real images.
- Whether near-duplicates or crops cross splits.
- Whether the dataset license permits the intended application or redistribution.
- Whether performance generalizes to video deepfakes or unseen datasets.

Kaggle metadata lists the dataset as CC BY-NC-SA 4.0. Whether that license and
any rights related to depicted people permit a specific deployment or model
redistribution still requires review. See
[`docs/dataset_provenance.md`](dataset_provenance.md).

## Reproducible command

With the saved manifest:

```sh
python3 research/audit_split.py \
  "/path/to/truthtrail_data/dataset_split_seed42.json"
```

If the original dataset is available locally, add `--dataset-root` pointing to
the directory that directly contains `training_real/` and `training_fake/` to
check exact file-content hashes across splits. The script does not download or
modify data.

## Recommended research follow-up

Obtain a documented source/group identifier for each image, or recreate a
source-disjoint split from the original data. Then repeat evaluation and save
the new manifest and per-image predictions before making generalization claims.
