"""Leakage-aware semantic-feature ablation for Complaint Intelligence.

The experiment compares classical features, direct text embeddings and optional
LLM-summary embeddings. Preprocessing is fitted on training data only, model and
scenario selection use validation F1, and the held-out test split is evaluated
once for the selected training fit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import clean_data  # noqa: E402
from train import split_data  # noqa: E402


CAT_COLS = [
    "product", "sub_product", "issue", "sub_issue", "state",
    "tags", "is_weekend", "zip_regional", "company_cleaned",
]
NUM_COLS = [
    "days_to_send", "narrative_length", "narrative_word_count",
    "mask_count", "exclamation_count", "question_count",
    "received_month", "received_day_of_week",
]


def add_research_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    text = result["complaint_what_happened"].fillna("").astype(str)

    result["mask_count"] = text.str.upper().str.count("XXXX").astype(float)
    result["exclamation_count"] = text.str.count("!").astype(float)
    result["question_count"] = text.str.count(r"\?").astype(float)

    received_raw = result["date_received"] if "date_received" in result else pd.Series(pd.NaT, index=result.index)
    sent_raw = result["date_sent_to_company"] if "date_sent_to_company" in result else pd.Series(pd.NaT, index=result.index)
    received = pd.to_datetime(received_raw, errors="coerce")
    sent = pd.to_datetime(sent_raw, errors="coerce")
    result["days_to_send"] = (sent - received).dt.days.fillna(0).clip(lower=0).astype(float)
    result["received_month"] = received.dt.month.fillna(0).astype(int)
    result["received_day_of_week"] = received.dt.dayofweek.fillna(0).astype(int)
    result["is_weekend"] = np.where(result["received_day_of_week"] >= 5, "Yes", "No")

    zip_code = result.get("zip_code", pd.Series("00000", index=result.index)).fillna("00000").astype(str)
    result["zip_regional"] = zip_code.str[:3]

    company = result.get("company", pd.Series("Unknown", index=result.index)).fillna("Unknown").astype(str)
    result["company_cleaned"] = company.str.upper().str.strip()

    if "tags" not in result:
        result["tags"] = "Missing"
    result["tags"] = result["tags"].fillna("Missing").astype(str)
    return result


def group_companies_from_train(train_df: pd.DataFrame, *other_frames: pd.DataFrame):
    """Learn the frequent-company grouping from training data only."""
    train = train_df.copy()
    others = [frame.copy() for frame in other_frames]
    top_companies = set(train["company_cleaned"].value_counts().head(10).index)

    train["company_cleaned"] = train["company_cleaned"].where(
        train["company_cleaned"].isin(top_companies), "OTHER COMPANIES"
    )
    for frame in others:
        frame["company_cleaned"] = frame["company_cleaned"].where(
            frame["company_cleaned"].isin(top_companies), "OTHER COMPANIES"
        )
    return (train, *others)


def prepare_classical(train_df: pd.DataFrame, *other_frames: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    frames = [frame.copy() for frame in (train_df, *other_frames)]
    for frame in frames:
        for col in CAT_COLS:
            if col not in frame:
                frame[col] = "Missing"
            frame[col] = frame[col].fillna("Missing").astype(str)
        for col in NUM_COLS:
            if col not in frame:
                frame[col] = 0.0
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )
    train_features = preprocessor.fit_transform(frames[0])
    return (train_features, *(preprocessor.transform(frame) for frame in frames[1:]))


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_embedder(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def encode(embedder, texts: pd.Series) -> np.ndarray:
    return embedder.encode(
        texts.fillna("").astype(str).tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def load_qwen(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if use_cuda:
        model = model.to(device)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if use_cuda else -1,
    )


def qwen_summary(generator, text: str) -> str:
    prompt = (
        "Summarize the consumer complaint in one short factual sentence. "
        "Do not add facts that are not present in the complaint.\n\nComplaint:\n"
        + text[:2500]
    )
    output = generator(prompt, max_new_tokens=70, do_sample=False, return_full_text=False)
    return str(output[0]["generated_text"]).replace("\n", " ").strip()


def fit_pca(train_embeddings: np.ndarray, *other_embeddings: np.ndarray, components: int):
    from sklearn.decomposition import PCA

    n_components = min(components, train_embeddings.shape[0], train_embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    train_reduced = pca.fit_transform(train_embeddings)
    return (train_reduced, *(pca.transform(values) for values in other_embeddings))


def metrics(model, features, target) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    pred = model.predict(features)
    prob = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(target, pred)),
        "precision": float(precision_score(target, pred, zero_division=0)),
        "recall": float(recall_score(target, pred, zero_division=0)),
        "f1_score": float(f1_score(target, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, prob)),
    }


def model_candidates():
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "XGBoost": lambda: XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
        ),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-aware S1-S4 complaint representation ablations.")
    parser.add_argument("--data", type=Path, required=True, help="Local CFPB-style complaint CSV.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/research_ablation.csv"))
    parser.add_argument("--device", default="auto", help="Embedding/LLM device: auto, cuda, cuda:0 or cpu.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--pca-components", type=int, default=15)
    parser.add_argument("--summary-column", default="llm_summary")
    parser.add_argument("--generate-summaries", action="store_true")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    raw = pd.read_csv(args.data, dtype={"complaint_id": str})
    cleaned = add_research_features(clean_data(raw))
    train_df, validation_df, test_df = split_data(cleaned, seed=42)
    train_df, validation_df, test_df = group_companies_from_train(
        train_df, validation_df, test_df
    )

    x_train_classic, x_val_classic, x_test_classic = prepare_classical(
        train_df, validation_df, test_df
    )

    device = resolve_device(args.device)
    embedder = build_embedder(args.embedding_model, device)
    direct_train = encode(embedder, train_df["complaint_what_happened"])
    direct_val = encode(embedder, validation_df["complaint_what_happened"])
    direct_test = encode(embedder, test_df["complaint_what_happened"])
    direct_train_pca, direct_val_pca, direct_test_pca = fit_pca(
        direct_train, direct_val, direct_test, components=args.pca_components
    )

    scenarios = {
        "S1: Classical Baseline": (x_train_classic, x_val_classic, x_test_classic),
        "S2: Classical + Direct Text Embedding": (
            np.hstack([x_train_classic, direct_train_pca]),
            np.hstack([x_val_classic, direct_val_pca]),
            np.hstack([x_test_classic, direct_test_pca]),
        ),
    }

    if args.generate_summaries:
        generator = load_qwen(args.qwen_model, device)
        for frame in (train_df, validation_df, test_df):
            frame[args.summary_column] = [
                qwen_summary(generator, text)
                for text in frame["complaint_what_happened"].astype(str)
            ]

    if all(args.summary_column in frame.columns for frame in (train_df, validation_df, test_df)):
        summary_train = encode(embedder, train_df[args.summary_column])
        summary_val = encode(embedder, validation_df[args.summary_column])
        summary_test = encode(embedder, test_df[args.summary_column])
        summary_train_pca, summary_val_pca, summary_test_pca = fit_pca(
            summary_train, summary_val, summary_test, components=args.pca_components
        )
        scenarios["S3: Classical + LLM Summary Embedding"] = (
            np.hstack([x_train_classic, summary_train_pca]),
            np.hstack([x_val_classic, summary_val_pca]),
            np.hstack([x_test_classic, summary_test_pca]),
        )
        scenarios["S4: Classical + Direct Text + LLM Summary Embedding"] = (
            np.hstack([x_train_classic, direct_train_pca, summary_train_pca]),
            np.hstack([x_val_classic, direct_val_pca, summary_val_pca]),
            np.hstack([x_test_classic, direct_test_pca, summary_test_pca]),
        )

    y_train = train_df["is_relief"].to_numpy()
    y_validation = validation_df["is_relief"].to_numpy()
    y_test = test_df["is_relief"].to_numpy()

    validation_results = []
    fitted = {}
    for scenario_name, (x_train, x_validation, x_test) in scenarios.items():
        for model_name, factory in model_candidates().items():
            model = factory()
            model.fit(x_train, y_train)
            result = {
                "split": "validation",
                "model_name": model_name,
                "scenario": scenario_name,
                **metrics(model, x_validation, y_validation),
            }
            validation_results.append(result)
            fitted[(model_name, scenario_name)] = (model, x_test)

    winner = max(validation_results, key=lambda row: row["f1_score"])
    winner_key = (winner["model_name"], winner["scenario"])
    winner_model, winner_test_features = fitted[winner_key]
    test_result = {
        "split": "test",
        "model_name": winner["model_name"],
        "scenario": winner["scenario"],
        **metrics(winner_model, winner_test_features, y_test),
    }

    output_rows = validation_results + [test_result]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output, index=False)
    print(json.dumps({"selected_on_validation": winner, "held_out_test": test_result}, indent=2))


if __name__ == "__main__":
    main()
