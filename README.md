# FLAG face–voice verification | English protocol-specific KCCA system

Organizers reviewing CodaBench submission 1054878 can find the v428 checkpoints and replay instructions in [v428 reference](reference/README.md). The rest of this README describes the local KCCA candidate.

This is the English-trained candidate retained by the project on 19 November. On five held-out folds of official English identities it achieved 16.604% standard EER and 27.216% same-gender EER. The mean was 21.910%. The package was evaluated locally. Its official four-condition English–Bangla EER is unknown. [Two-page system description](documents/系统说明.pdf) · [Per-fold results](kcca-reference/verified_metrics.json)

Each protocol averages a fixed neural scoring ensemble and an ensemble trained with protocol-specific negative pairs. A regularized radial basis kernel canonical correlation model adds a calibrated score from training-identity centers. The deployed system loads four neural parameter files and five per-fold kernel files. The English-only selection used a prespecified kernel. An earlier multilingual model reached 21.790% on the English trials under a different training setup. Across held-out sources, the candidate improved the English–Urdu four-condition average from 31.038% to 29.967% against the cross-video baseline. The two seven-person Bangla same-gender checks moved in opposite directions. [Paired evaluations](EDA/11-19-公开原声与跨来源更新/output/11-19-核方向跨语逐对验证/核方向与分协议结果.json)

## Verification

The development metrics can be recalculated from this checkout alone once dependencies are installed. The frozen checkpoint replay additionally needs eleven external files consisting of the official trial lists, row indexes and five feature arrays. Their relative paths and SHA-256 digests are listed under the replay stage of the resource inventory. Licensed per-media features and raw media must be made available separately. [Resource inventory](resources.json)

The recorded environment was Linux and Python 3.11 in Conda environment `my_project_env`. Running the original training entry points uses CUDA. For a fresh installation use the included lock and CUDA index with these commands.

```bash
conda create -n my_project_env python=3.11
conda activate my_project_env
uv export --frozen --extra train --no-dev --no-emit-project --emit-index-url --format requirements-txt --output-file /tmp/flag-kcca-requirements.txt
uv pip install --python "$CONDA_PREFIX/bin/python" -r /tmp/flag-kcca-requirements.txt
python reproduce.py metrics
python reproduce.py replay --resources /path/to/authorized-tree --output /tmp/flag-english-kcca.zip
```

The replay verifies all nine scoring parameter files before inference and compares the output archive with the original local candidate. The archive contains 392 standard English, 802 standard Bangla, 416 same-gender English and 716 same-gender Bangla trials. The expected ZIP SHA-256 is `faf1c779c4a86b3dbe5b1a9c09ac5a5c35f3e348c9ffc0f5526afbf5bfcd2ec3`. The release launcher adds configurable feature and output paths and loads the optional external validation suite only when requested. The score transformation and decimal serialization follow the archived implementation. [Verification record](kcca-reference/verified_replay.json)

## Original fitting stages

The original directory layout preserves source scripts, study configurations, fitting and calibration splits, input hashes and saved models. The fixed neural models, protocol-specific models and kernel models each have a different fitting entry. The table lists execution order and deployment artifacts.

| Stage | Training and export |
| --- | --- |
| Fixed neural models | Run `train.py`, `ensemble.py` and `official_stage.py` in `EDA/11-18-英语训练池从头对照/` in sequence. Export the protocol parameters with `english_shared.py` and `english_remove_eres.py` in `EDA/11-18-跨录制协方差迁移/`. The standard branch removes the ERes2NetV2 contribution. |
| Protocol-specific neural models | Run `production.py` in `EDA/11-18-跨语言协议训练/`. It creates a timestamped output directory. Run `EDA/11-18-Vox英语身份扩充/evaluate.py` with each new arm directory (`count_matched`, `same_gender`) as its positional argument to fit the scoring parameters. |
| Kernel models | Run `anchor_train.py` in `EDA/11-19-公开原声与跨来源更新/`, supplying a new output directory as its positional argument. The original runner fits the development, source-holdout and control models. The deployed scorer uses the five development-fold RBF files. |
| Raw inputs | The original v379–v383 extraction scripts cover the frozen image and audio models. The Vox selection code, threshold settings and selected-source record are archived in `EDA/11-18-Vox英语身份扩充/`. Obtain the media through the source datasets and the frontend checkpoints from their providers. |

Use a separate working copy for historical training scripts that write to fixed output directories. Reconstruct the dataset layout and restore the marked external files before fitting. The historical diagnostics also read protected source media, raw identity metadata and extra validation feature sets, including CREMA-D and Bangla. Some historical configurations and media indexes store absolute paths from the original machine. The supplied development scores and the complete frozen-parameter ZIP were independently replayed. Full retraining from raw media on a new host has not been verified. [Training configuration](EDA/11-19-公开原声与跨来源更新/英语训练人物核对齐配置.json) · [Selection before evaluation](EDA/11-19-公开原声与跨来源更新/output/11-19-核方向跨语逐对验证/评分前锁定.json) · [Record of the original package](kcca-reference/archive_audit.json)
