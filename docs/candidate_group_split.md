# Candidate grouped split

## Why this exists

The saved manifest has 2,041 rows: 1,081 real and 960 fake. Every fake
filename's middle number, 1 through 480, also appears as a real filename ID
after normalizing leading zeros. The dataset README describes the fake-name
parts as indicating replaced face regions, but does not say that this number
identifies an original image or person. This alignment is therefore a
candidate grouping signal, not a confirmed source identifier.

The proposal below is a sensitivity-analysis artifact. It does not replace the
seed-42 manifest and must not be used for model evaluation until the filename
semantics have been confirmed. No model was trained or evaluated with it.

## Provisional assignment

Using the shared-number assumption, the manifest forms 1,081 candidate groups
with a maximum of four rows per group. A deterministic greedy assignment
(seed 42) keeps all inferred groups together and targets the current split
proportions and class counts:

| Split | Original real | Original fake | Proposed real | Proposed fake | Proposed total |
|---|---:|---:|---:|---:|---:|
| Train | 756 | 672 | 756 | 692 | 1,448 |
| Validation | 162 | 144 | 162 | 134 | 296 |
| Test | 163 | 144 | 163 | 134 | 297 |

The proposed assignment has zero groups crossing splits under the assumed
numeric grouping rule. Its class counts are close to the existing split, but
this does not establish that the groups represent shared source images or
people.

## Reproduce the report

The script prints the candidate counts without writing a file:

```sh
python3 research/propose_grouped_split.py \
  "/path/to/truthtrail_data/dataset_split_seed42.json"
```

To explicitly accept the unconfirmed assumption and write a separate
candidate manifest, provide a new output path:

```sh
python3 research/propose_grouped_split.py \
  "/path/to/truthtrail_data/dataset_split_seed42.json" \
  --accept-filename-link-assumption \
  --output "/path/to/candidate_grouped_split.json"
```

The script refuses to overwrite an existing file or the source manifest. The
generated candidate is not a replacement for the official split. Confirm the
dataset filename convention with its provider, or obtain documented source
IDs, before using grouped partitions for evaluation.
