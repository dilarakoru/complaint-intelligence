# Historical research modules
These modules preserve the earlier XGBoost/LightGBM, embeddings, PCA and SHAP work for review.
They are **not** the validated serving or evaluation pipeline. `run_pipeline.py` is retained for historical context, not as a supported command.
The original experiment contained train/test duplicates and selected its best model on test F1.
Do not cite the old scores. No old model, cache, personal narratives or evaluation artifacts are committed.
Use `../train.py` for the corrected baseline. The optional on-device LLM summary is a separate feature and does not claim to improve classification accuracy.


## Semantic embedding ablation

`semantic_ablation.py` is a cleaned reference implementation of the historical S1-S4 experiment:

- S1 — classical structured features;
- S2 — classical features + direct complaint-text embedding;
- S3 — classical features + Qwen summary embedding;
- S4 — classical features + direct embedding + Qwen summary embedding.

It uses `all-MiniLM-L6-v2`, PCA, XGBoost/LightGBM and optionally `Qwen/Qwen2.5-0.5B-Instruct`. Unlike the original notebook experiment, this public script reuses the repository's current duplicate cleaning and leakage-aware split logic.

Install the optional research stack with:

```bash
pip install -r research/requirements-experiments.txt
```

Run direct-embedding experiments:

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

Historical saved metrics are documented in [../docs/EXPERIMENT_HISTORY.md](../docs/EXPERIMENT_HISTORY.md). They should not be treated as results from this cleaned script unless the same experiment is rerun and recorded.
