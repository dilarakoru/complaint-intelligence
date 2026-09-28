# Experiment history

This document records earlier Complaint Intelligence experiments that are separate from the current supported CPU-compatible serving/evaluation pipeline.

## Compute and language-model experiments

A historical Colab notebook records CUDA execution on an **NVIDIA L4** GPU. That work used:

- `Qwen/Qwen2.5-0.5B-Instruct` through Hugging Face Transformers;
- SentenceTransformer embeddings with `all-MiniLM-L6-v2`;
- PCA for dimensionality reduction;
- XGBoost and LightGBM classifiers;
- SHAP for feature-importance analysis.

The notebook explored four representation strategies:

- S1 — classical/tabular baseline;
- S2 — classical features + direct text embedding;
- S3 — classical features + LLM summary embedding;
- S4 — classical features + direct embedding + LLM summary embedding.

## Evaluation lesson

The historical experiment produced apparently stronger scores, but those values are **not valid generalization evidence**. The old workflow contained train/test duplicate contamination and compared candidate models/scenarios on test F1. Those issues make the recorded scores unsuitable as a benchmark, so this repository intentionally does not present them as current performance claims.

The useful conclusion from that work is methodological: richer representations and an LLM-derived summary were worth exploring, but representation complexity must be evaluated with an isolated validation set and a genuinely untouched test set.

The cleaned public implementation in [../research/semantic_ablation.py](../research/semantic_ablation.py) therefore:

1. reuses the current duplicate/conflict cleaning;
2. creates train/validation/test splits with identity and normalized-text isolation;
3. learns category grouping, preprocessing and PCA on training data only;
4. selects the model/scenario using validation F1;
5. evaluates the selected training fit once on held-out test data.

No cleaned GPU ablation score is claimed until that corrected experiment is rerun and its inputs are documented.

## Why the current application is simpler

The public application intentionally uses a smaller, inspectable scikit-learn pipeline so that data cleaning, leakage controls, model selection, API behavior and explanations can be reproduced locally without requiring GPU infrastructure.
