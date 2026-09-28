import os
import pickle
import numpy as np
import pandas as pd
import json
from data_loader import fetch_complaints_data, preprocess_and_prepare_dataset
from llm_processor import LLMProcessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from explainability import SHAPExplainer

def main():
    print("======================================================================")
    # 0. Define paths
    dir_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(dir_path, exist_ok=True)
    raw_path = os.path.join(dir_path, "complaints_raw.csv")
    processed_path = os.path.join(dir_path, "complaints_processed.csv")
    pipeline_obj_path = os.path.join(dir_path, "pipeline_objects.pkl")
    
    # 1. Fetch data if not cached
    if not os.path.exists(raw_path):
        # Fetch 500 records (balanced for training speed and LLM removal of noise)
        fetch_complaints_data(raw_path, target_size=500)
        
    print("Regenerating processed dataset to apply latest cleaning steps...")
    preprocess_and_prepare_dataset(raw_path, processed_path, missing_rate=0.10)
            
    df = pd.read_csv(processed_path)
    print(f"Loaded processed dataset. Shape: {df.shape}")
    
    # Veri doğrulama ve hedef değişken istatistikleri
    print("\n--- Veri Doğrulama ve Hedef Değişken İstatistikleri ---")
    print(f"Toplam Gözlem Sayısı: {len(df)}")
    print(f"Hedef Değişken ('is_relief') Dağılımı:")
    dist = df["is_relief"].value_counts(normalize=True)
    counts = df["is_relief"].value_counts()
    for val, p in dist.items():
        label = "Telafi Sağlandı (1)" if val == 1 else "Açıklama/Red (0)"
        print(f"  - {label}: {counts[val]} adet ({p*100:.2f}%)")
    
    print("\nKategorik Alanlarda Yapay Eksik Değer Oranları:")
    for col in ["sub_product", "sub_issue"]:
        na_count = df[col].isna().sum()
        print(f"  - {col}: {na_count} eksik hücre ({na_count/len(df)*100:.2f}%)")
    
    print(f"\nSeçilen Sosyo-Demografik Öznitelik ('tags') Dağılımı:")
    print(df["tags"].value_counts())
    print("----------------------------------------------------\n")
    
    # 2. Split dataset into train and test
    fe = FeatureEngineer(pca_components=15)
    train_df, test_df = fe.split_data(df)
    print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
    
    # 3. LLM Processing (Imputation & Semantic Analysis)
    llm = LLMProcessor()
    # Loading models will trigger downloads if not already cached
    llm.load_models()
    
    # Get imputation choices from train set
    sub_product_choices = [c for c in train_df["sub_product_true"].unique() if pd.notna(c)]
    sub_issue_choices = [c for c in train_df["sub_issue_true"].unique() if pd.notna(c)]
    
    # Phase 2: LLM Context-Aware Imputation
    print("\n--- Phase 2: Performing LLM Context-Aware Imputation ---")
    
    for name, dataset in [("Train", train_df), ("Test", test_df)]:
        print(f"Imputing missing values in {name} dataset...")
        
        # Sub-product
        imputed_sub_products = []
        for idx, row in dataset.iterrows():
            if pd.isna(row["sub_product"]):
                val = llm.impute_missing_value(row.to_dict(), "sub_product", sub_product_choices)
                imputed_sub_products.append(val)
            else:
                imputed_sub_products.append(row["sub_product"])
        dataset["sub_product_llm"] = imputed_sub_products
        
        # Sub-issue
        imputed_sub_issues = []
        for idx, row in dataset.iterrows():
            if pd.isna(row["sub_issue"]):
                val = llm.impute_missing_value(row.to_dict(), "sub_issue", sub_issue_choices)
                imputed_sub_issues.append(val)
            else:
                imputed_sub_issues.append(row["sub_issue"])
        dataset["sub_issue_llm"] = imputed_sub_issues
        
    print("LLM Imputation completed.")
    
    # Phase 1: Semantic Analysis and Gömüler (Embeddings)
    print("\n--- Phase 1: Performing LLM Semantic Analysis on narratives ---")
    for name, dataset in [("Train", train_df), ("Test", test_df)]:
        print(f"Analyzing narrative text for {name} dataset...")
        summaries = []
        for idx, row in dataset.iterrows():
            summary = llm.generate_semantic_analysis(
                row["complaint_what_happened"], 
                row["product"], 
                row["issue"]
            )
            summaries.append(summary)
        dataset["complaint_summary"] = summaries
        
    print("Semantic analysis generated. Creating Sentence-Transformer Embeddings...")
    train_embeddings = llm.get_embeddings(train_df["complaint_summary"].tolist())
    test_embeddings = llm.get_embeddings(test_df["complaint_summary"].tolist())
    print(f"Train embeddings shape: {train_embeddings.shape}, Test shape: {test_embeddings.shape}")
    
    # Save datasets with LLM results for backup and reference
    train_df.to_csv(os.path.join(dir_path, "train_llm_processed.csv"), index=False)
    test_df.to_csv(os.path.join(dir_path, "test_llm_processed.csv"), index=False)
    
    # 4. Preparing configurations for the 4 scenarios
    print("\n--- Preparing 4 Scenarios ---")
    
    # --- Scenario 1: Classical + Mode Imputation
    # Prepare classical imputed sets
    train_s1_raw, test_s1_raw = fe.impute_classical(train_df, test_df)
    X_train_s1, X_test_s1 = fe.prepare_classical_features(train_s1_raw, test_s1_raw, fit=True)
    # Backup feature transformer from Scenario 1 to apply to Scenario 2
    preprocessor_s1 = fe.preprocessor
    
    # --- Scenario 2: Classical + LLM Imputation
    # Map LLM imputed columns to the classical names to pass through the same preprocessor
    train_s2_raw = train_df.copy().drop(columns=["sub_product", "sub_issue"]).rename(columns={"sub_product_llm": "sub_product", "sub_issue_llm": "sub_issue"})
    test_s2_raw = test_df.copy().drop(columns=["sub_product", "sub_issue"]).rename(columns={"sub_product_llm": "sub_product", "sub_issue_llm": "sub_issue"})
    fe.preprocessor = preprocessor_s1 # Reuse preprocessor mapping to ensure shape compatibility
    X_train_s2, X_test_s2 = fe.prepare_classical_features(train_s2_raw, test_s2_raw, fit=False)
    
    # --- Scenario 3: Hybrid + Mode Imputation
    # Apply PCA on embeddings
    pca_train, pca_test = fe.apply_pca_on_embeddings(train_embeddings, test_embeddings, fit=True)
    X_train_s3 = fe.create_hybrid_space(X_train_s1, pca_train)
    X_test_s3 = fe.create_hybrid_space(X_test_s1, pca_test)
    
    # --- Scenario 4: Hybrid + LLM Imputation
    X_train_s4 = fe.create_hybrid_space(X_train_s2, pca_train)
    X_test_s4 = fe.create_hybrid_space(X_test_s2, pca_test)
    
    y_train = train_df["is_relief"].values
    y_test = test_df["is_relief"].values
    
    data_dict = {
        "scenario_1": (X_train_s1, X_test_s1, y_train, y_test),
        "scenario_2": (X_train_s2, X_test_s2, y_train, y_test),
        "scenario_3": (X_train_s3, X_test_s3, y_train, y_test),
        "scenario_4": (X_train_s4, X_test_s4, y_train, y_test)
    }
    
    # 5. Train and compare models
    trainer = ModelTrainer()
    best_model, best_metrics, results_df = trainer.run_all_scenarios(data_dict)
    
    # 6. Fit Explainer on the best scenario features
    print("\n--- SHAP Explainer Calculation ---")
    best_scenario = best_metrics["scenario"]
    
    # Extract training and testing features for the best scenario
    if "S1:" in best_scenario:
        best_X_train, best_X_test = X_train_s1, X_test_s1
        best_features = fe.feature_names
    elif "S2:" in best_scenario:
        best_X_train, best_X_test = X_train_s2, X_test_s2
        best_features = fe.feature_names
    elif "S3:" in best_scenario:
        best_X_train, best_X_test = X_train_s3, X_test_s3
        best_features = fe.feature_names + [f"PCA_Embed_{i+1}" for i in range(fe.pca_components)]
    else: # S4
        best_X_train, best_X_test = X_train_s4, X_test_s4
        best_features = fe.feature_names + [f"PCA_Embed_{i+1}" for i in range(fe.pca_components)]
        
    explainer = SHAPExplainer(best_model, best_features)
    explainer.fit_explainer(best_X_train)
    explainer.calculate_shap(best_X_test)
    
    # Save static summary plot
    shap_plot_path = os.path.join(dir_path, "shap_summary.png")
    explainer.save_summary_plot(best_X_test, save_path=shap_plot_path)
    
    # Save global feature importance JSON
    global_imp = explainer.get_global_importance()
    with open(os.path.join(dir_path, "shap_global_importance.json"), "w") as f:
        json.dump(global_imp, f, indent=2)
        
    print(f"SHAP calculations saved. Best features count: {len(best_features)}")
    
    # --- Statistics and Quality Metrics Calculations ---
    # Calculate imputation metrics for evaluation
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_s1_df = pd.concat([train_s1_raw, test_s1_raw], ignore_index=True)
    
    # sub_product imputation accuracy
    sub_prod_masked = full_df["sub_product"].isna() & full_df["sub_product_true"].notna()
    if sub_prod_masked.sum() > 0:
        true_sub_prod = full_df.loc[sub_prod_masked, "sub_product_true"]
        llm_sub_prod = full_df.loc[sub_prod_masked, "sub_product_llm"]
        mode_sub_prod = full_s1_df.loc[sub_prod_masked, "sub_product"]
        
        llm_sub_prod_acc = float((true_sub_prod == llm_sub_prod).mean())
        mode_sub_prod_acc = float((true_sub_prod == mode_sub_prod).mean())
    else:
        llm_sub_prod_acc = 0.0
        mode_sub_prod_acc = 0.0
        
    # sub_issue imputation accuracy
    sub_issue_masked = full_df["sub_issue"].isna() & full_df["sub_issue_true"].notna()
    if sub_issue_masked.sum() > 0:
        true_sub_issue = full_df.loc[sub_issue_masked, "sub_issue_true"]
        llm_sub_issue = full_df.loc[sub_issue_masked, "sub_issue_llm"]
        mode_sub_issue = full_s1_df.loc[sub_issue_masked, "sub_issue"]
        
        llm_sub_issue_acc = float((true_sub_issue == llm_sub_issue).mean())
        mode_sub_issue_acc = float((true_sub_issue == mode_sub_issue).mean())
    else:
        llm_sub_issue_acc = 0.0
        mode_sub_issue_acc = 0.0
        
    imputation_stats = {
        "sub_product_mode_acc": mode_sub_prod_acc,
        "sub_product_llm_acc": llm_sub_prod_acc,
        "sub_issue_mode_acc": mode_sub_issue_acc,
        "sub_issue_llm_acc": llm_sub_issue_acc,
        "sub_product_masked_count": int(sub_prod_masked.sum()),
        "sub_issue_masked_count": int(sub_issue_masked.sum())
    }
    
    with open(os.path.join(dir_path, "imputation_stats.json"), "w") as f:
        json.dump(imputation_stats, f, indent=2)
    print("\n--- Imputation Accuracy Stats ---")
    print(f"sub_product - Mode Accuracy: {mode_sub_prod_acc:.4f}, LLM Accuracy: {llm_sub_prod_acc:.4f}")
    print(f"sub_issue   - Mode Accuracy: {mode_sub_issue_acc:.4f}, LLM Accuracy: {llm_sub_issue_acc:.4f}")

    # Calculate EDA and data stats for charts in the dashboard
    class_dist_counts = df["is_relief"].value_counts().to_dict()
    class_dist_pct = df["is_relief"].value_counts(normalize=True).to_dict()
    
    class_dist = {
        "0": {"count": int(class_dist_counts.get(0, 0)), "pct": float(class_dist_pct.get(0, 0.0))},
        "1": {"count": int(class_dist_counts.get(1, 0)), "pct": float(class_dist_pct.get(1, 0.0))}
    }
    
    top_products = df["product"].value_counts().head(8).to_dict()
    top_companies = df["company_cleaned"].value_counts().head(8).to_dict()
    
    narrative_lengths = df["narrative_length"].describe().to_dict()
    narrative_words = df["narrative_word_count"].describe().to_dict()
    mask_counts = df["mask_count"].describe().to_dict()
    
    missing_rates = {
        "sub_product": float(df["sub_product"].isna().mean()),
        "sub_issue": float(df["sub_issue"].isna().mean())
    }
    
    # Confusion matrix for the best model
    from sklearn.metrics import confusion_matrix
    y_pred_best = best_model.predict(best_X_test)
    cm = confusion_matrix(y_test, y_pred_best).tolist() # [[tn, fp], [fn, tp]]
    
    # PCA explained variance
    pca_variance = []
    if fe.pca is not None and hasattr(fe.pca, "explained_variance_ratio_"):
        pca_variance = fe.pca.explained_variance_ratio_.tolist()
        
    data_stats = {
        "class_distribution": class_dist,
        "top_products": top_products,
        "top_companies": top_companies,
        "narrative_length_stats": {k: float(v) for k, v in narrative_lengths.items()},
        "narrative_word_stats": {k: float(v) for k, v in narrative_words.items()},
        "mask_count_stats": {k: float(v) for k, v in mask_counts.items()},
        "missing_rates": missing_rates,
        "confusion_matrix": cm,
        "pca_explained_variance": pca_variance
    }
    
    with open(os.path.join(dir_path, "data_stats.json"), "w") as f:
        json.dump(data_stats, f, indent=2)
    print("EDA and model statistics saved to data_stats.json.")
    
    # 7. Save pipeline objects for inference in web dashboard
    pipeline_objects = {
        "best_model": best_model,
        "best_scenario": best_scenario,
        "best_metrics": best_metrics.to_dict(),
        "preprocessor": preprocessor_s1,
        "pca": fe.pca,
        "pca_components": fe.pca_components,
        "feature_names": fe.feature_names,
        "best_features": best_features,
        "sub_product_choices": sub_product_choices,
        "sub_issue_choices": sub_issue_choices,
        "is_hybrid": "Hybrid" in best_scenario,
        "X_train_sample": best_X_train,
        "imputation_stats": imputation_stats,
        "data_stats": data_stats
    }
    
    with open(pipeline_obj_path, "wb") as f:
        pickle.dump(pipeline_objects, f)
    print(f"Pipeline objects saved to {pipeline_obj_path}.")
    print("Pipeline execution complete! Ready for Web Dashboard.")

if __name__ == "__main__":
    main()
