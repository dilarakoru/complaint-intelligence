# Research modules

The supported application and evaluation path lives at the repository root. This folder contains optional semantic-feature experiments and historical utilities; none of them are used by the FastAPI serving path.

## Semantic embedding ablation

`semantic_ablation.py` is the cleaned reference implementation of the S1-S4 experiment:

- S1 — classical structured features;
- S2 — classical features + direct complaint-text embedding;
- S3 — classical features + Qwen summary embedding;
- S4 — classical features + direct embedding + Qwen summary embedding.

It uses `all-MiniLM-L6-v2`, PCA, XGBoost/LightGBM and optionally `Qwen/Qwen2.5-0.5B-Instruct`. The script reuses the repository's current duplicate cleaning and leakage-aware split logic, selects on validation F1 and evaluates the selected training fit once on held-out test data.

Install the optional research stack with:

```bash
pip install -r research/requirements-experiments.txt
```

Run direct-embedding experiments:

```bash
python research/semantic_ablation.py --data /path/to/complaints.csv
```

Use CUDA when available:

```bash
python research/semantic_ablation.py --data /path/to/complaints.csv --device cuda
```

To generate Qwen summaries and include S3/S4:

```bash
python research/semantic_ablation.py \
  --data /path/to/complaints.csv \
  --device cuda \
  --generate-summaries
```

The original notebook experiment contained train/test duplicates and selected its best candidate on test F1. Its scores are retained only in the private research history and are not valid benchmark evidence. See [../docs/EXPERIMENT_HISTORY.md](../docs/EXPERIMENT_HISTORY.md).

`feature_engineer.py` and `explainability.py` preserve earlier PCA/feature-engineering and SHAP exploration for code review. They are not imported by the serving application.


## Historical split overlap audit

`audit_split_overlap.py` reproduces the identity/text overlap check when the original private train/test CSVs are available:

```bash
python research/audit_split_overlap.py \
  --train /path/to/old_train.csv \
  --test /path/to/old_test.csv
```

The public repository does not include the original complaint narratives, so the historical 66/80 ID-overlap and 68/80 normalized-text-overlap counts cannot be recomputed from the synthetic fixture. The current pipeline instead enforces zero overlap through assertions and automated tests.
