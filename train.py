"""Train/validation/test evaluation with identity and normalized-text isolation."""
import argparse
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_loader import clean_data

ROOT = Path(__file__).resolve().parent
CAT = ['product', 'sub_product', 'issue', 'sub_issue', 'company']
NUM = ['narrative_length', 'narrative_word_count']

def split_data(df, seed=42):
    if len(df) < 30 or df.is_relief.value_counts().min() < 6 or df.is_relief.nunique() != 2:
        raise ValueError('Need at least 30 unique records and 6 records per class.')
    train, holdout = train_test_split(df, test_size=.4, random_state=seed, stratify=df.is_relief)
    val, test = train_test_split(holdout, test_size=.5, random_state=seed, stratify=holdout.is_relief)
    for a, b in [(train, val), (train, test), (val, test)]:
        for key in ['complaint_id', '_text_group']:
            if set(a[key]) & set(b[key]): raise AssertionError(f'Leakage in {key}')
    return train, val, test

def build_model(use_text):
    columns = [('category', OneHotEncoder(handle_unknown='ignore'), CAT),
               ('numeric', StandardScaler(), NUM)]
    if use_text:
        columns.append(('text', TfidfVectorizer(min_df=1, max_features=2000, ngram_range=(1,2), sublinear_tf=True), 'complaint_what_happened'))
    return Pipeline([('features', ColumnTransformer(columns)),
                     ('classifier', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42))])

def metrics(model, df):
    y = df.is_relief; pred = model.predict(df); p = model.predict_proba(df)[:, 1]
    return {'accuracy': float(accuracy_score(y, pred)), 'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)), 'f1': float(f1_score(y, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, p)), 'confusion_matrix': confusion_matrix(y, pred, labels=[0,1]).tolist()}

def train(input_path, output_dir, seed=42, synthetic=False):
    raw = pd.read_csv(input_path, dtype={'complaint_id':str})
    df = clean_data(raw); tr, val, test = split_data(df, seed)
    models = {}; validation = {}
    for name, use_text in [('structured_baseline', False), ('text_hybrid', True)]:
        models[name] = build_model(use_text).fit(tr, tr.is_relief)
        validation[name] = metrics(models[name], val)
    winner = max(validation, key=lambda name: validation[name]['f1'])
    # Keep the selected training fit unchanged; inspect test only after selection.
    final = models[winner]
    report = {'data_kind':'synthetic demonstration' if synthetic else 'local research dataset',
              'input_sha256':hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
              'raw_rows':len(raw), 'unique_clean_rows':len(df), 'removed_rows':len(raw)-len(df),
              'split_sizes':{'train':len(tr),'validation':len(val),'test':len(test)},
              'seed':seed, 'id_overlap':0, 'normalized_text_overlap':0,
              'validation':validation, 'selected_model':winner, 'test':metrics(final,test),
              'limitations':['Small non-representative sample; not a production performance estimate.',
                             'TF-IDF is a classical text baseline, not an LLM.',
                             'This experiment replaces the original leaked evaluation; results are not directly comparable.']}
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    with (output/'model.pkl').open('wb') as f:pickle.dump(final,f)
    (output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/demo.csv')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--synthetic',action='store_true');a=p.parse_args()
    train(a.data,a.output,a.seed,a.synthetic or a.data.resolve()==(ROOT/'data/demo.csv').resolve())
