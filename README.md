# Complaint Intelligence

> Product walkthrough, design trade-offs and next experiments: [Engineering notes](docs/ENGINEERING.md).

![Application preview](docs/preview.png)

A local-first NLP product prototype: submit a complaint, inspect a classification with feature contributions, and ask a small on-device language model to select a grounded source excerpt.

The central engineering story is **finding and fixing data leakage**, not claiming a high accuracy score. The previous experiment mixed duplicate complaints across training and test data. This version isolates identities and normalized text, selects its model on validation data, and reports a separate test result.

## Quick start

Tested with Python 3.10. Create a separate environment for this project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001. `train.py` defaults to the committed **synthetic** dataset; its scores only verify the pipeline. No account or paid API is required.

On macOS/Linux, activate with `source .venv/bin/activate` instead. Always use Python 3.10 for the validated pinned environment.

## Optional local LLM

```powershell
python -m pip install -r requirements-llm.txt
$env:ENABLE_LOCAL_LLM='1'
$env:ALLOW_MODEL_DOWNLOAD='1'  # first use only, downloads Qwen weights
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

The model is `Qwen/Qwen2.5-0.5B-Instruct`. It runs on CPU. Once weights are cached, unset `ALLOW_MODEL_DOWNLOAD` for offline use. Loading is lazy; startup and classification do not need the LLM.

The model selects an exact source excerpt rather than freely inventing a summary. Generated text must be present verbatim in the input after whitespace normalization. If validation fails, the first source sentence is returned with an explicit fallback label. This constrains unsupported claims; it does not guarantee that the chosen sentence is the most useful one.

## Architecture

```text
CSV -> deterministic cleaning/deduplication -> 60/20/20 stratified split
    -> training-only preprocessing -> structured vs TF-IDF hybrid baseline
    -> validation F1 model selection -> held-out test -> local artifact

Browser -> FastAPI -> classifier + exact linear feature contributions
                   -> optional Qwen -> source-excerpt validation/fallback
```

- `data_loader.py`: rejects conflicting duplicate groups and removes duplicate IDs/text.
- `train.py`: compares two logistic regression pipelines with train-only category/text vocabularies.
- `app.py`: typed prediction API, health endpoint, evaluation endpoint, local LLM endpoint.
- `llm_processor.py`: optional local inference and source grounding.
- `tests/`: leakage, conflicting labels, API, invalid input, and hallucinated excerpt checks.
- `research/`: historical XGBoost/LightGBM/PCA/SHAP modules for provenance; not the serving pipeline or a supported command.

The UI's feature values are **linear log-odds contributions**, not SHAP values. TF-IDF is a classical text representation, not an LLM. The optional LLM does not change classification results.

## Evaluation

The corrected local research run used 151 clean unique records from 400 source rows: 90 training, 30 validation, and 31 test records. Both identity and normalized-text overlap were zero. The validation-selected structured baseline obtained test F1 **0.5806**, accuracy **0.5806**, and ROC-AUC **0.6807**.

These are exploratory results on a very small, non-representative sample. They are not a production performance claim. The old 0.875 accuracy is invalid as independent evidence because of duplicate leakage; the old and new experiments also use different model setups, so their scores are not an apples-to-apples comparison. See [evaluation details](docs/evaluation-local.json).

Local data is excluded from Git. To reproduce with your own authorized data:

```powershell
python train.py --data data/local/complaints_raw.csv
```

Required columns: `complaint_id`, `complaint_what_happened`, `company_response`. Accepted targets are `Closed with explanation`, `Closed with monetary relief`, and `Closed with non-monetary relief`. Optional features: product, sub_product, issue, sub_issue, company. The public synthetic CSV documents the format.

## Test

```powershell
python -m unittest discover -s tests -v
```

Never load externally supplied pickle files. The application only reads the artifact generated locally by its training command, and does not expose its source/data directory as static content.

## Product limitations

This is a research demonstration, not an outcome guarantee or financial decision system. Classifier scores are not calibrated probabilities. The tiny dataset limits generalization. Near-duplicate paraphrases and temporal shifts require a larger future evaluation. The optional LLM is slow on CPU and may use the deterministic fallback. The application binds to localhost and has no public-service authentication/rate limiting.

## Portfolio preparation

This version adds safe dataset splitting, a reproducible synthetic demo, an independent test set, source-grounded LLM output, an API/UI, tests and run instructions. See [provenance](docs/PROVENANCE.md) for what is new versus historical research.
