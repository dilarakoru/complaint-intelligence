# Experiment history

This document records earlier Complaint Intelligence experiments that are separate from the current supported CPU-compatible serving/evaluation pipeline.

## Compute and language-model experiments

A historical Colab notebook records CUDA execution on an **NVIDIA L4** GPU. That work used:

- `Qwen/Qwen2.5-0.5B-Instruct` through Hugging Face Transformers;
- SentenceTransformer embeddings with `all-MiniLM-L6-v2`;
- PCA for dimensionality reduction;
- XGBoost and LightGBM classifiers;
- SHAP for feature-importance analysis.

The notebook compared four representation strategies:

- S1 — classical/tabular baseline;
- S2 — classical features + direct text embedding;
- S3 — classical features + LLM summary embedding;
- S4 — classical features + direct embedding + LLM summary embedding.

## Recorded ablation result

One saved experiment table contains the following comparison:

| Model / scenario | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost — S4 | 0.7987 | 0.8176 | 0.7683 | 0.7921 | 0.8470 |
| XGBoost — S2 | 0.7924 | 0.8087 | 0.7651 | 0.7863 | 0.8512 |
| LightGBM — S2 | 0.7845 | 0.7954 | 0.7651 | 0.7799 | 0.8463 |
| XGBoost — S1 | 0.7829 | 0.7908 | 0.7683 | 0.7794 | 0.8540 |
| XGBoost — S3 | 0.7829 | 0.8027 | 0.7492 | 0.7750 | 0.8553 |

The experiment is useful because it shows that adding an LLM-derived summary was not automatically the strongest representation: the direct-embedding configuration remained competitive and produced the highest ROC-AUC among the rows shown above.

These historical results are not the benchmark for the current deployed classifier. They come from a separate experiment setup and are documented as development history rather than current product performance.

## Why the current application is simpler

The public application intentionally uses a smaller, inspectable scikit-learn pipeline so that data cleaning, leakage controls, model selection, API behavior and explanations can be reproduced locally without requiring GPU infrastructure.