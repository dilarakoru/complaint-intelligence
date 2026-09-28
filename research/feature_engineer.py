import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class FeatureEngineer:
    def __init__(self, pca_components=15, test_size=0.2, seed=42):
        self.pca_components = pca_components
        self.test_size = test_size
        self.seed = seed
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_components, random_state=self.seed)
        
        self.preprocessor = None
        self.cat_cols = ["product", "sub_product", "issue", "sub_issue", "state", "tags", "is_weekend", "zip_regional", "company_cleaned"]
        self.num_cols = ["days_to_send", "narrative_length", "narrative_word_count", "mask_count", "exclamation_count", "question_count", "received_month", "received_day_of_week"]

    def split_data(self, df):
        """Splits the dataframe into train and test sets using stratified split."""
        # For tiny datasets, fall back to non-stratified split to avoid errors
        stratify_col = df["is_relief"] if len(df) >= 10 else None
        train_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            random_state=self.seed, 
            stratify=stratify_col
        )
        return train_df.copy(), test_df.copy()

    def impute_classical(self, train_df, test_df):
        """Imputes missing categorical values using the most frequent value (Mode)."""
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        for col in ["sub_product", "sub_issue"]:
            mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else "Missing"
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)
            
        return train_df, test_df

    def prepare_classical_features(self, train_df, test_df, fit=True):
        """
        Scales numerical features and one-hot encodes categorical features.
        Returns dense feature arrays.
        """
        # Ensure state and other categorical columns are string, and fill missing state/zip values
        for df in [train_df, test_df]:
            for col in self.cat_cols:
                df[col] = df[col].fillna("Missing").astype(str)
            for col in self.num_cols:
                df[col] = df[col].fillna(0.0)

        if fit:
            # Categorical Pipeline: standard One-Hot Encoder
            # handle_unknown='ignore' handles categories present in test but not train
            cat_pipeline = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            
            # Numerical Pipeline: Standard Scaler
            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, self.num_cols),
                    ("cat", cat_pipeline, self.cat_cols)
                ]
            )
            
            X_train_processed = self.preprocessor.fit_transform(train_df)
        else:
            X_train_processed = self.preprocessor.transform(train_df)
            
        X_test_processed = self.preprocessor.transform(test_df)
        
        # Get feature names for interpretability
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = cat_encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + cat_feature_names
        
        return X_train_processed, X_test_processed

    def apply_pca_on_embeddings(self, train_embeddings, test_embeddings, fit=True):
        """Fits PCA on training embeddings and transforms both training and test embeddings."""
        if fit:
            pca_features_train = self.pca.fit_transform(train_embeddings)
        else:
            pca_features_train = self.pca.transform(train_embeddings)
            
        pca_features_test = self.pca.transform(test_embeddings)
        return pca_features_train, pca_features_test

    def create_hybrid_space(self, X_classical, X_pca_embeddings):
        """
        Concatenates classical features and PCA-reduced semantic embeddings.
        X_hybrid = X_num_cat_encoded + PCA(X_sem)
        """
        return np.hstack((X_classical, X_pca_embeddings))

if __name__ == "__main__":
    # Test script locally with dummy data
    print("Testing Feature Engineer...")
    fe = FeatureEngineer(pca_components=2)
    
    # Create simple dummy data
    data = {
        "complaint_id": [1, 2, 3, 4, 5],
        "product": ["Credit card", "Mortgage", "Credit card", "Debt collection", "Mortgage"],
        "sub_product": ["General card", np.nan, "General card", "Medical debt", np.nan],
        "issue": ["Billing", "Interest", "Billing", "Harassment", "Interest"],
        "sub_issue": ["Fees", np.nan, "Fees", "Calls", "Fees"],
        "state": ["CA", "NY", "CA", "TX", "NY"],
        "tags": ["None", "Servicemember", "None", "Older American", "None"],
        "days_to_send": [1.0, 2.0, 0.0, 4.0, 1.0],
        "narrative_length": [100.0, 200.0, 150.0, 300.0, 120.0],
        "narrative_word_count": [20.0, 40.0, 30.0, 60.0, 25.0],
        "mask_count": [0.0, 1.0, 0.0, 2.0, 0.0],
        "exclamation_count": [0.0, 0.0, 1.0, 0.0, 0.0],
        "question_count": [0.0, 0.0, 0.0, 1.0, 0.0],
        "received_month": [9.0, 9.0, 9.0, 9.0, 9.0],
        "received_day_of_week": [1.0, 2.0, 0.0, 4.0, 1.0],
        "is_weekend": ["No", "No", "Yes", "No", "Yes"],
        "company_cleaned": ["BANK OF AMERICA", "JPMORGAN CHASE", "OTHER COMPANIES", "WELLS FARGO", "BANK OF AMERICA"],
        "zip_regional": ["902", "100", "941", "770", "100"],
        "is_relief": [1, 0, 1, 0, 0]
    }
    df = pd.DataFrame(data)
    
    train_df, test_df = fe.split_data(df)
    print("Train size:", len(train_df), "Test size:", len(test_df))
    
    train_imp, test_imp = fe.impute_classical(train_df, test_df)
    print("Classical Imputed sub_product values:\n", train_imp["sub_product"])
    
    X_train_c, X_test_c = fe.prepare_classical_features(train_imp, test_imp)
    print("Classical processed train features shape:", X_train_c.shape)
    
    # Dummy embeddings
    dummy_train_embed = np.random.rand(len(train_df), 384)
    dummy_test_embed = np.random.rand(len(test_df), 384)
    
    pca_train, pca_test = fe.apply_pca_on_embeddings(dummy_train_embed, dummy_test_embed)
    print("PCA train features shape:", pca_train.shape)
    
    X_train_h = fe.create_hybrid_space(X_train_c, pca_train)
    print("Hybrid train features shape:", X_train_h.shape)
