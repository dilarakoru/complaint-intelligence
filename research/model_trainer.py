import os
import json
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelTrainer:
    def __init__(self, seed=42):
        self.seed = seed
        self.results = []
        
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name, scenario_name):
        """Trains a model and returns its performance metrics."""
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5  # Fallback for single class in split
            
        result = {
            "model_name": model_name,
            "scenario": scenario_name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc)
        }
        return result, model

    def run_all_scenarios(self, data_dict):
        """
        Runs and evaluates all 4 scenarios for both XGBoost and LightGBM.
        data_dict contains:
          - scenario_1: (X_train, X_test, y_train, y_test)
          - scenario_2: (X_train, X_test, y_train, y_test)
          - scenario_3: (X_train, X_test, y_train, y_test)
          - scenario_4: (X_train, X_test, y_train, y_test)
        """
        self.results = []
        trained_models = {}
        
        scenarios = {
            "S1: Baseline (Classic + Mode Imp)": data_dict["scenario_1"],
            "S2: Classic + LLM Imp": data_dict["scenario_2"],
            "S3: Hybrid (Classic + PCA_Embed + Mode Imp)": data_dict["scenario_3"],
            "S4: Hybrid + LLM Imp": data_dict["scenario_4"]
        }
        
        for scenario_name, (X_train, X_test, y_train, y_test) in scenarios.items():
            print(f"\n--- Running Training for Scenario: {scenario_name} ---")
            print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
            
            # 1. XGBoost
            xgb = XGBClassifier(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.05, 
                random_state=self.seed,
                eval_metric="logloss"
            )
            xgb_metrics, xgb_model = self.evaluate_model(xgb, X_train, y_train, X_test, y_test, "XGBoost", scenario_name)
            self.results.append(xgb_metrics)
            trained_models[f"XGBoost_{scenario_name}"] = xgb_model
            print(f"XGBoost F1: {xgb_metrics['f1_score']:.4f} | AUC: {xgb_metrics['roc_auc']:.4f}")
            
            # 2. LightGBM
            lgb = LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.seed,
                verbosity=-1
            )
            lgb_metrics, lgb_model = self.evaluate_model(lgb, X_train, y_train, X_test, y_test, "LightGBM", scenario_name)
            self.results.append(lgb_metrics)
            trained_models[f"LightGBM_{scenario_name}"] = lgb_model
            print(f"LightGBM F1: {lgb_metrics['f1_score']:.4f} | AUC: {lgb_metrics['roc_auc']:.4f}")
            
        # Compile results
        results_df = pd.DataFrame(self.results)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        results_df.to_json(os.path.join(BASE_DIR, "model_results.json"), orient="records", indent=2)
        print("\nAll Scenarios Run. Results saved to model_results.json.")
        print(results_df.to_string(index=False))
        
        # Find best model based on F1-score (or ROC-AUC)
        best_row = results_df.sort_values(by="f1_score", ascending=False).iloc[0]
        best_key = f"{best_row['model_name']}_{best_row['scenario']}"
        best_model = trained_models[best_key]
        
        print(f"\nBest Model: {best_key} with F1-score: {best_row['f1_score']:.4f}")
        return best_model, best_row, results_df

if __name__ == "__main__":
    # Test locally with dummy shapes
    print("Testing Model Trainer...")
    trainer = ModelTrainer()
    
    # Generate dummy data splits for all 4 scenarios
    dummy_dict = {
        "scenario_1": (np.random.rand(80, 10), np.random.rand(20, 10), np.random.randint(0, 2, 80), np.random.randint(0, 2, 20)),
        "scenario_2": (np.random.rand(80, 10), np.random.rand(20, 10), np.random.randint(0, 2, 80), np.random.randint(0, 2, 20)),
        "scenario_3": (np.random.rand(80, 15), np.random.rand(20, 15), np.random.randint(0, 2, 80), np.random.randint(0, 2, 20)),
        "scenario_4": (np.random.rand(80, 15), np.random.rand(20, 15), np.random.randint(0, 2, 80), np.random.randint(0, 2, 20))
    }
    
    best_m, best_r, res_df = trainer.run_all_scenarios(dummy_dict)
