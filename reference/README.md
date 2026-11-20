# FLAG 2027 inference

This package contains the 15 frozen model checkpoints and their 15 Gaussian scoring parameter files for replay of CodaBench entry **1054878**, which recorded a mean EER of **9.87%**. Training for this submission used supervision derived from joint scoring of Evaluation samples. The procedure is disclosed for compliance review. [Archived run record](weights/result.json)

## Run the replay

The replay uses Python 3.11 and the original PyTorch 2.8 CUDA implementation. Run the commands below from the repository root with the Conda environment `my_project_env` active. The archived environment used CUDA 12.8.

```bash
cd reference
uv export --frozen --extra train --no-dev --no-emit-project --emit-index-url \
  --format requirements-txt --output-file /tmp/flag2027-inference-requirements.txt
uv pip install --python "$CONDA_PREFIX/bin/python" -r /tmp/flag2027-inference-requirements.txt
python review.py replay --resources /path/to/authorized-tree --work /tmp/flag-replay
cat /tmp/flag-replay/comparison.json
```

Use a fresh `--work` path. The [resource inventory](resources.json) lists each input path and SHA-256 digest. The authorized resource tree must supply the eight inputs marked `"bundled": false`: one Evaluation trial manifest, two row indexes, and five feature archives. Those resources require separate authorized access. This package starts from precomputed features. The scoring script evaluates the same 15 heads for each trial and writes four score files. The replay command compares all 2,326 predictions with the archived reference files. The [local replay comparison](reference/replay-comparison.json) records a byte-for-byte match for all four score files.

[Checkpoints and scoring parameters](weights/) · [Scoring entry point](inference/package.py) · [CodaBench record](reference/entry.json)
