# Historical research modules
These modules preserve the earlier XGBoost/LightGBM, embeddings, PCA and SHAP work for review.
They are **not** the validated serving or evaluation pipeline. `run_pipeline.py` is retained for provenance, not a supported command.
The original experiment contained train/test duplicates and selected its best model on test F1.
Do not cite the old scores. No old model, cache, personal narratives or evaluation artifacts are committed.
Use `../train.py` for the corrected baseline. The optional on-device LLM summary is a separate feature and does not claim to improve classification accuracy.
