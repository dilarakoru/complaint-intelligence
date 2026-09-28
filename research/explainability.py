import os
import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def fit_explainer(self, X_train):
        """Initializes the SHAP Explainer using TreeExplainer with fallback to Explainer."""
        print("Fitting SHAP Explainer...")
        try:
            self.explainer = shap.TreeExplainer(self.model)
            print("Successfully initialized TreeExplainer.")
        except Exception as e:
            print(f"TreeExplainer failed ({e}). Falling back to model-agnostic Explainer...")
            # Use predict function directly to bypass model parsing
            predict_fn = self.model.predict
            self.explainer = shap.Explainer(predict_fn, X_train)
        return self.explainer

    def calculate_shap(self, X_eval):
        """Calculates SHAP values for evaluation dataset."""
        if self.explainer is None:
            self.fit_explainer(X_eval)
        
        # Calculate SHAP values
        if hasattr(self.explainer, "shap_values"):
            self.shap_values = self.explainer.shap_values(X_eval)
        else:
            explanation = self.explainer(X_eval)
            self.shap_values = explanation.values
        
        # In multi-class or binary model output shape checks:
        # For binary classification, shap_values might be a list of length 2 (one per class),
        # or a 3D array, or a 2D array. Let's make it consistent.
        if isinstance(self.shap_values, list):
            # Binary classification list: index 1 corresponds to class 1 (relief)
            if len(self.shap_values) == 2:
                self.shap_values = self.shap_values[1]
            else:
                self.shap_values = self.shap_values[0]
        elif len(self.shap_values.shape) == 3: # (n_samples, n_features, n_classes)
            if self.shap_values.shape[2] == 2:
                self.shap_values = self.shap_values[:, :, 1]
            else:
                self.shap_values = self.shap_values[:, :, 0]
                
        return self.shap_values

    def save_summary_plot(self, X_eval, save_path=None):
        """Saves a static SHAP summary plot using Matplotlib."""
        if save_path is None:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(BASE_DIR, "shap_summary.png")
            
        if self.shap_values is None:
            self.calculate_shap(X_eval)
            
        plt.figure(figsize=(10, 6))
        # Use pandas DataFrame to keep feature names clean on the plot
        X_df = pd.DataFrame(X_eval, columns=self.feature_names)
        shap.summary_plot(self.shap_values, X_df, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"SHAP summary plot saved to {save_path}.")

    def get_global_importance(self):
        """Returns the mean absolute SHAP values for all features sorted."""
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first.")
            
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": mean_abs_shap
        }).sort_values(by="importance", ascending=False)
        
        return importance_df.to_dict(orient="records")

    def get_local_explanation(self, X_single, feature_values):
        """
        Explains a single prediction row and returns list of dicts.
        X_single: 1D numpy array of processed features
        feature_values: 1D array of original or processed readable values
        """
        if self.explainer is None:
            self.fit_explainer(X_single.reshape(1, -1))
            
        if hasattr(self.explainer, "shap_values"):
            single_shap = self.explainer.shap_values(X_single.reshape(1, -1))
        else:
            explanation = self.explainer(X_single.reshape(1, -1))
            single_shap = explanation.values
        
        if isinstance(single_shap, list):
            if len(single_shap) == 2:
                single_shap = single_shap[1]
            else:
                single_shap = single_shap[0]
        elif len(single_shap.shape) == 3: # (n_samples, n_features, n_classes)
            if single_shap.shape[2] == 2:
                single_shap = single_shap[:, :, 1]
            else:
                single_shap = single_shap[:, :, 0]
                
        single_shap = single_shap.flatten()
        
        local_df = pd.DataFrame({
            "feature": self.feature_names,
            "value": [str(v) for v in feature_values],
            "shap_value": [float(sv) for sv in single_shap]
        })
        
        # Sort by absolute SHAP value
        local_df["abs_shap"] = local_df["shap_value"].abs()
        local_df = local_df.sort_values(by="abs_shap", ascending=False).drop(columns=["abs_shap"])
        
        # Get base value
        base_value = 0.0
        if hasattr(self.explainer, "expected_value"):
            base_value = self.explainer.expected_value
        elif hasattr(self.explainer, "expected_values"): # e.g. for some explainers
            base_value = self.explainer.expected_values
            
        if isinstance(base_value, (list, np.ndarray)):
            if len(base_value) == 2:
                base_value = float(base_value[1])
            else:
                base_value = float(base_value[0])
        else:
            base_value = float(base_value)
            
        return {
            "base_value": base_value,
            "features": local_df.to_dict(orient="records")
        }

if __name__ == "__main__":
    # Test script locally with dummy classifier
    print("Testing SHAP Explainer...")
    from xgboost import XGBClassifier
    
    # Simple training data
    X_train = np.random.rand(50, 4)
    y_train = np.random.randint(0, 2, 50)
    feature_names = ["days_to_send", "narrative_length", "PCA_1", "PCA_2"]
    
    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)
    
    explainer = SHAPExplainer(model, feature_names)
    explainer.fit_explainer(X_train)
    
    shap_vals = explainer.calculate_shap(X_train[:10])
    print("SHAP values shape:", shap_vals.shape)
    
    importance = explainer.get_global_importance()
    print("Top feature:", importance[0])
    
    local_exp = explainer.get_local_explanation(X_train[0], X_train[0])
    print("Local explanation base value:", local_exp["base_value"])
    print("Local explanation first feature contribution:", local_exp["features"][0])
    
    # Save test summary plot
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    explainer.save_summary_plot(X_train[:10], save_path=os.path.join(BASE_DIR, "shap_summary_test.png"))
