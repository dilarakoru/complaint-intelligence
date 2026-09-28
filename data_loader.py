"""Deterministic cleaning. No fitted statistics are learned before splitting."""
import re
import pandas as pd

def normalize_text(value):
    return re.sub(r"\s+", " ", str(value).casefold()).strip()

def clean_data(raw):
    needed = {'complaint_id', 'complaint_what_happened', 'company_response'}
    if not needed.issubset(raw.columns):
        raise ValueError(f'Missing columns: {sorted(needed - set(raw.columns))}')
    df = raw.copy()
    valid = ['Closed with explanation', 'Closed with monetary relief', 'Closed with non-monetary relief']
    df = df[df.company_response.isin(valid)].copy()
    df['complaint_what_happened'] = df.complaint_what_happened.fillna('').astype(str)
    df = df[df.complaint_what_happened.str.strip().ne('')].copy()
    df['_text_group'] = df.complaint_what_happened.map(normalize_text)
    # Remove ambiguous duplicate groups rather than arbitrarily selecting a target.
    conflict_ids = df.groupby('complaint_id').company_response.nunique()
    conflict_texts = df.groupby('_text_group').company_response.nunique()
    df = df[~df.complaint_id.isin(conflict_ids[conflict_ids > 1].index)]
    df = df[~df._text_group.isin(conflict_texts[conflict_texts > 1].index)]
    df = df.drop_duplicates('complaint_id').drop_duplicates('_text_group').copy()
    df['is_relief'] = df.company_response.str.contains('relief', regex=False).astype(int)
    for col in ['product', 'sub_product', 'issue', 'sub_issue', 'state', 'company']:
        if col not in df: df[col] = 'Unknown'
        df[col] = df[col].fillna('Unknown').astype(str).str.strip().replace('', 'Unknown')
    df['narrative_length'] = df.complaint_what_happened.str.len()
    df['narrative_word_count'] = df.complaint_what_happened.str.split().str.len()
    return df.reset_index(drop=True)
