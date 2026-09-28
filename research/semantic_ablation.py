"""Reproducible semantic-feature ablation for Complaint Intelligence.

This is a cleaned research script based on the historical GPU notebook. It
preserves the S1-S4 comparison (classical features, direct text embeddings and
LLM-summary embeddings) while reusing the current repository's leakage-aware
cleaning and split logic.

Heavy research dependencies are intentionally isolated from the production app.
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
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "state",
    "tags",
    "is_weekend",
    "zip_regional",
    "company_cleaned",
]
NUM_COLS = [
    "days_to_send",
    "narrative_length",
    "narrative_word_count",
    "mask_count",
    "exclamation_count",
    "question_count",
    "received_month",
    "received_day_of_week",
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
    top = set(result["company_cleaned"].value_counts().head(10).index)
    result["company_cleaned"] = result["company_cleaned"].where(
        result["company_cleaned"].isin(top), "OTHER COMPANIES"
    )

    if "tags" not in result:
        result["tags"] = "Missing"
    result["tags"] = result["tags"].fillna("Missing").astype(str)
    return result


def classical_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    tr = train_df.copy()
    te = test_df.copy()
    for col in CAT_COLS:
        if col not in tr:
            tr[col] = "Missing"
        if col not in te:
            te[col] = "Missing"
        tr[col] = tr[col].fillna("Missing").astype(str)
        te[col] = te[col].fillna("Missing").astype(str)

    for col in NUM_COLS:
        if col not in tr:
            tr[col] = 0.0
        if col not in te:
            te[col] = 0.0
        tr[col] = pd.to_numeric(tr[col], errors="coerce").fillna(0.0)
        te[col] = pd.to_numeric(te[col], errors="coerce").fillna(0.0)

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )
    return preprocessor.fit_transform(tr), preprocessor.transform(te)


def embed(texts: list[str], model_name: str, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)


def load_qwen(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def qwen_summary(generator, text: str) -> str:
    prompt = (
        "Summarize the consumer complaint in one short factual sentence. "
        "Do not add facts that are not present in the complaint.\n\nComplaint:\n"
        + text[:2500]
    )
    output = generator(prompt, max_new_tokens=70, do_sample=False, return_full_text=False)
    generated = output[0]["generated_text"]
    return str(generated).replace("\n", " ").strip()


def reduce_pca(train_embeddings: np.ndarray, test_embeddings: np.ndarray, components: int):
    from sklearn.decomposition import PCA

    n_components = min(components, train_embeddings.shape[0], train_embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(train_embeddings), pca.transform(test_embeddings)


def evaluate(model, x_train, x_test, y_train, y_test, model_name: str, scenario: str) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, 1]
    return {
        "model_name": model_name,
        "scenario": scenario,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S1-S4 complaint representation ablations.")
    parser.add_argument("--data", type=Path, required=True, help="Local CFPB-style complaint CSV.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/research_ablation.csv"))
    parser.add_argument("--device", default="cuda", help="SentenceTransformer device, e.g. cuda or cpu.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--pca-components", type=int, default=15)
    parser.add_argument("--summary-column", default="llm_summary")
    parser.add_argument("--generate-summaries", action="store_true")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    raw = pd.read_csv(args.data, dtype={"complaint_id": str})
    cleaned = add_research_features(clean_data(raw))
    train_df, _, test_df = split_data(cleaned, seed=42)

    x_train_classic, x_test_classic = classical_features(train_df, test_df)
    direct_train = embed(
        train_df["complaint_what_happened"].astype(str).tolist(),
        args.embedding_model,
        args.device,
    )
    direct_test = embed(
        test_df["complaint_what_happened"].astype(str).tolist(),
        args.embedding_model,
        args.device,
    )
    direct_train_pca, direct_test_pca = reduce_pca(
        direct_train, direct_test, args.pca_components
    )

    scenarios = {
        "S1: Classical Baseline": (x_train_classic, x_test_classic),
        "S2: Classical + Direct Text Embedding": (
            np.hstack([x_train_classic, direct_train_pca]),
            np.hstack([x_test_classic, direct_test_pca]),
        ),
    }

    if args.generate_summaries:
        generator = load_qwen(args.qwen_model)
        train_df[args.summary_column] = [
            qwen_summary(generator, text)
            for text in train_df["complaint_what_happened"].astype(str)
        ]
        test_df[args.summary_column] = [
            qwen_summary(generator, text)
            for text in test_df["complaint_what_happened"].astype(str)
        ]

    if args.summary_column in train_df.columns and args.summary_column in test_df.columns:
        summary_train = embed(
            train_df[args.summary_column].fillna("").astype(str).tolist(),
            args.embedding_model,
            args.device,
        )
        summary_test = embed(
            test_df[args.summary_column].fillna("").astype(str).tolist(),
            args.embedding_model,
            args.device,
        )
        summary_train_pca, summary_test_pca = reduce_pca(
            summary_train, summary_test, args.pca_components
        )
        scenarios["S3: Classical + LLM Summary Embedding"] = (
            np.hstack([x_train_classic, summary_train_pca]),
            np.hstack([x_test_classic, summary_test_pca]),
        )
        scenarios["S4: Classical + Direct Text + LLM Summary Embedding"] = (
            np.hstack([x_train_classic, direct_train_pca, summary_train_pca]),
            np.hstack([x_test_classic, direct_test_pca, summary_test_pca]),
        )

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    y_train = train_df["is_relief"].to_numpy()
    y_test = test_df["is_relief"].to_numpy()
    results = []
    for scenario, (x_train, x_test) in scenarios.items():
        results.append(
            evaluate(
                XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    eval_metric="logloss",
                ),
                x_train,
                x_test,
                y_train,
                y_test,
                "XGBoost",
                scenario,
            )
        )
        results.append(
            evaluate(
                LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    verbosity=-1,
                ),
                x_train,
                x_test,
                y_train,
                y_test,
                "LightGBM",
                scenario,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).sort_values("f1_score", ascending=False).to_csv(
        args.output, index=False
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
