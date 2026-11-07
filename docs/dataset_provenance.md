# Dataset provenance and license review

Checked on 2026-11-07. This is a research note, not legal advice.

## Identified dataset

The project manifest and notebook point to the **Real and Fake Face
Detection** dataset published by CIPLAB @ Yonsei University:

- [Kaggle dataset page](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
- [Dataset source repository and README](https://github.com/minostauros/Real-and-Fake-Face-Detection)
- Kaggle metadata identifies version 1, creator CIPLAB @ Yonsei University,
  and license **CC BY-NC-SA 4.0**.

The source README describes expert-created Photoshop composites and says the
`easy`, `mid`, and `hard` fake categories are subjective. It also says
filenames describe which face parts were replaced. That documentation does not
establish that a filename number identifies a person, source photograph, or
manipulation lineage.

## License considerations

The [CC BY-NC-SA 4.0 deed](https://creativecommons.org/licenses/by-nc-sa/4.0/)
allows sharing and adaptation subject to attribution, non-commercial use, and
ShareAlike requirements. It also notes that other rights, including privacy
and publicity rights, may limit a particular use.

Before redistributing images, derived datasets, trained weights, or deploying
the application beyond a non-commercial academic demonstration, review the
full license and confirm the intended use with the instructor, dataset
provider, or an appropriate rights adviser. This note does not decide whether
the model weights are a derivative work or what obligations apply to them.

## Local evidence and audit limits

The available research files are outside the application repository:

- `truthtrail_data/dataset_split_seed42.json`
- `truthtrail_data/results/`
- `archive_01/mltruthtrail-2.ipynb`

The split manifest has 1,428 train, 306 validation, and 307 test rows. Each row
contains only `path` and `label`; the paths point to a Colab `/content/dataset/`
location. The prediction CSVs contain paths, labels, scores, and predictions,
but no source-person, original-image, or group identifiers. The dataset images
were not found in the local research folder, so image hashes and source
relationships could not be checked directly.

The read-only filename audit found 153 numeric-token groups crossing splits.
This is a review signal only. The source README explains filename parts as
replacement-region information, and the available metadata does not establish
that the numeric token is a source identity. It is therefore not evidence of
confirmed leakage.

## Next research action

Obtain authorized access to the original files and any source/pairing metadata
from the dataset provider. If no reliable source identifiers exist, record
that limitation and avoid describing the current image-level split as
source-disjoint. Do not download or redistribute the dataset until the
applicable terms and project approval are clear.
