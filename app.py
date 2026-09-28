import json
import os
import pickle
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from llm_processor import summarize

ROOT=Path(__file__).resolve().parent
ARTIFACTS=Path(os.getenv('COMPLAINT_ARTIFACTS',str(ROOT/'artifacts')))
app=FastAPI(title='Complaint Intelligence',version='1.0.0')
class Complaint(BaseModel):
    text:str=Field(min_length=20,max_length=10000)
    product:str=Field(default='Credit card',max_length=200)
    issue:str=Field(default='Billing dispute',max_length=200)
    sub_product:str=Field(default='Unknown',max_length=200)
    sub_issue:str=Field(default='Unknown',max_length=200)
    company:str=Field(default='Unknown',max_length=200)

@lru_cache(maxsize=1)
def get_model():
    path=ARTIFACTS/'model.pkl'
    if not path.exists():raise HTTPException(503,'Model missing. Run python train.py first.')
    # Load only a locally trained artifact, never an uploaded pickle.
    with path.open('rb') as f:return pickle.load(f)

@app.get('/',response_class=HTMLResponse)
def index():return (ROOT/'templates/index.html').read_text(encoding='utf-8')

@app.get('/health')
def health():return {'status':'ok','model_ready':(ARTIFACTS/'model.pkl').exists(),'llm_enabled':os.getenv('ENABLE_LOCAL_LLM')=='1'}

@app.get('/api/evaluation')
def evaluation():
    p=ARTIFACTS/'evaluation.json'
    if not p.exists():raise HTTPException(503,'Train the model first.')
    return json.loads(p.read_text(encoding='utf-8'))

@app.post('/api/predict')
def predict(req:Complaint):
    model=get_model()
    row=req.model_dump();row['complaint_what_happened']=row.pop('text')
    row['narrative_length']=len(row['complaint_what_happened']);row['narrative_word_count']=len(row['complaint_what_happened'].split())
    df=pd.DataFrame([row]);prob=float(model.predict_proba(df)[0,1])
    matrix=model.named_steps['features'].transform(df)
    values=matrix.toarray()[0] if hasattr(matrix,'toarray') else np.asarray(matrix)[0]
    effects=values*model.named_steps['classifier'].coef_[0]
    names=model.named_steps['features'].get_feature_names_out()
    important=np.argsort(np.abs(effects))[-8:][::-1]
    contributions=[{'feature':str(names[i]),'log_odds_contribution':float(effects[i])} for i in important if abs(effects[i])>1e-9]
    return {'prediction':'relief' if prob>=.5 else 'explanation','probability':prob,'contributions':contributions,
            'note':'Research demo; model score is not a calibrated probability or an individual outcome guarantee.'}

@app.post('/api/summarize')
def summary(req:Complaint):
    try:return {**summarize(req.text),'model':'Qwen2.5-0.5B-Instruct'}
    except Exception as exc:
        raise HTTPException(503,'Local summary unavailable. Enable the optional LLM and prepare its local weights.') from exc
