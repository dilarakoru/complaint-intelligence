import unittest
import tempfile
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
from data_loader import clean_data
from train import split_data, train, ROOT
import app as service

class PipelineTests(unittest.TestCase):
    def test_summary_rejects_invented_facts(self):
        from llm_processor import ground_summary
        source='A fee appeared on my statement. I requested a correction.'
        rejected=ground_summary(source,'The company confirmed that a refund was issued.')
        self.assertEqual(rejected['summary'],'A fee appeared on my statement.')
        self.assertIn('fallback',rejected['method'])
        accepted=ground_summary(source,'I requested a correction.')
        self.assertEqual(accepted['method'],'llm-selected source excerpt')

    def test_duplicates_cannot_cross_splits(self):
        raw=pd.read_csv(ROOT/'data/demo.csv')
        raw=pd.concat([raw,raw.iloc[:30]],ignore_index=True)
        df=clean_data(raw);parts=split_data(df)
        self.assertEqual(len(df),len(pd.read_csv(ROOT/'data/demo.csv')))
        for i,a in enumerate(parts):
            for b in parts[i+1:]:
                for key in ['complaint_id','_text_group']:
                    self.assertFalse(set(a[key])&set(b[key]))
    def test_conflicting_duplicates_are_excluded(self):
        raw=pd.read_csv(ROOT/'data/demo.csv');bad=raw.iloc[[0]].copy()
        bad['company_response']='Closed with explanation' if raw.iloc[0].company_response!='Closed with explanation' else 'Closed with monetary relief'
        self.assertEqual(len(clean_data(pd.concat([raw,bad]))),len(raw)-1)
    def test_research_company_grouping_uses_train_only(self):
        from research.semantic_ablation import group_companies_from_train
        train = pd.DataFrame({"company_cleaned": ["A"] * 4 + ["B"] * 3})
        validation = pd.DataFrame({"company_cleaned": ["A", "UNSEEN"]})
        grouped_train, grouped_validation = group_companies_from_train(train, validation)
        self.assertEqual(grouped_train.company_cleaned.tolist(), train.company_cleaned.tolist())
        self.assertEqual(grouped_validation.company_cleaned.tolist(), ["A", "OTHER COMPANIES"])

    def test_training_and_api(self):
        with tempfile.TemporaryDirectory() as folder:
            report=train(ROOT/'data/demo.csv',folder,synthetic=True)
            self.assertEqual(sum(report['split_sizes'].values()),report['unique_clean_rows'])
            old=service.ARTIFACTS;service.ARTIFACTS=Path(folder);service.get_model.cache_clear()
            try:
                with TestClient(service.app) as client:
                    self.assertEqual(client.get('/').status_code,200)
                    result=client.post('/api/predict',json={'text':'An unexpected fee appeared on my card and I requested a correction.'})
                    self.assertEqual(result.status_code,200);self.assertGreaterEqual(result.json()['probability'],0)
                    self.assertLessEqual(result.json()['probability'],1)
                    self.assertEqual(client.post('/api/predict',json={'text':'short'}).status_code,422)
                    self.assertEqual(client.get('/static/data/local/complaints_raw.csv').status_code,404)
                    self.assertEqual(client.post('/api/summarize',json={'text':'An unexpected fee appeared on my card and I requested a correction.'}).status_code,503)
            finally:service.ARTIFACTS=old;service.get_model.cache_clear()
if __name__=='__main__':unittest.main()
