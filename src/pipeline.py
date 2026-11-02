"""Reproducible binary-classification workflow; configure against the assigned dataset."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))

def load_config(path):
    cfg = json.loads(Path(path).read_text())
    for key in ['dataset', 'source', 'target', 'positive_label', 'expected_columns', 'drop_columns']:
        if key not in cfg:
            raise ValueError(f'Missing configuration field: {key}')
    if not cfg['expected_columns'] or not cfg['source']:
        raise ValueError('Provide the documented expected columns and dataset source.')
    return cfg

def validate(cfg):
    path = (ROOT / cfg['dataset']).resolve()
    if not path.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('The dataset must be loaded from the repository data folder.')
    df = pd.read_csv(path)
    if df.empty or not df.columns.is_unique:
        raise ValueError('Dataset must be nonempty with unique column names.')
    missing = set(cfg['expected_columns']) - set(df.columns)
    extra = set(df.columns) - set(cfg['expected_columns'])
    if missing or extra:
        raise ValueError(f'Schema mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}')
    target = cfg['target']
    if target not in df or df[target].isna().any() or df[target].nunique() != 2:
        raise ValueError('Target must contain exactly two classes and no missing labels.')
    if cfg['positive_label'] not in df[target].unique():
        raise ValueError('positive_label is absent from the target.')
    if target in cfg['drop_columns'] or set(cfg['drop_columns']) - set(df):
        raise ValueError('Invalid drop_columns configuration.')
    for col in cfg.get('numeric_columns', []):
        df[col] = pd.to_numeric(df[col], errors='raise')
        if np.isinf(df[col].dropna()).any():
            raise ValueError(f'Infinite values in {col}')
    summary = {'rows': len(df), 'columns': list(df), 'missing': df.isna().sum().to_dict(),
               'duplicate_rows': int(df.duplicated().sum()),
               'class_counts': df[target].value_counts().to_dict(),
               'source': cfg['source'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return df, summary

def prepare(cfg, out):
    df, summary = validate(cfg)
    save_json(out / 'registration.json', summary)
    print(json.dumps(summary, indent=2, default=str))
    # Remove IDs and post-outcome leakage fields before deduplication and splitting.
    df = df.drop(columns=cfg['drop_columns']).drop_duplicates()
    target = cfg['target']
    if len(df.columns) < 2:
        raise ValueError('No predictors remain.')
    if df.drop(columns=target).isna().all().any():
        raise ValueError('Remove entirely empty predictors in the configuration.')
    if df.groupby(list(df.drop(columns=target)), dropna=False)[target].nunique().gt(1).any():
        raise ValueError('Identical predictors have conflicting labels; resolve before splitting.')
    df[target] = (df[target] == cfg['positive_label']).astype(int)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    if train[target].value_counts().min() < 3:
        raise ValueError('Training data needs at least three observations in each class.')
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / 'train.csv', index=False)
    test.to_csv(out / 'test.csv', index=False)
    save_json(out / 'split_metadata.json', {'train_rows': len(train), 'test_rows': len(test),
        'removed_rows': summary['rows'] - len(df), 'seed': 42, 'target': target,
        'positive_label': cfg['positive_label'], 'drop_columns': cfg['drop_columns'],
        'source_sha256': summary['sha256'],
        'split_sha256': {name: hashlib.sha256((out / f'{name}.csv').read_bytes()).hexdigest()
                         for name in ['train', 'test']}})
    return train, test

def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {'average_precision': average_precision_score(y, prob), 'roc_auc': roc_auc_score(y, prob),
            'balanced_accuracy': balanced_accuracy_score(y, pred),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0), 'f1': f1_score(y, pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}

def train_model(cfg, splits, out):
    metadata = json.loads((splits / 'split_metadata.json').read_text())
    for key in ['target', 'positive_label', 'drop_columns']:
        if metadata[key] != cfg[key]:
            raise ValueError(f'Split artifact configuration mismatch: {key}')
    for name, digest in metadata['split_sha256'].items():
        if hashlib.sha256((splits / f'{name}.csv').read_bytes()).hexdigest() != digest:
            raise ValueError(f'Split artifact checksum mismatch: {name}')
    train, test = (pd.read_csv(splits / f'{name}.csv') for name in ['train', 'test'])
    target = cfg['target']
    X, y = train.drop(columns=target), train[target]
    Xt, yt = test.drop(columns=target), test[target]
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in X if c not in numeric]
    preprocessing = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical)])
    model = Pipeline([('preprocessing', preprocessing),
                      ('model', RandomForestClassifier(random_state=42, n_jobs=1))])
    grid = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10],
            'model__min_samples_leaf': [1, 3], 'model__class_weight': [None, 'balanced']}
    search = GridSearchCV(model, grid, scoring='average_precision',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=4,
        return_train_score=True, error_score='raise')
    search.fit(X, y)
    baseline = DummyClassifier(strategy='prior').fit(X, y)
    metrics = {'baseline': evaluate(baseline, Xt, yt), 'random_forest': evaluate(search.best_estimator_, Xt, yt),
               'best_cv_average_precision': search.best_score_, 'best_parameters': search.best_params_,
               'test_prevalence': float(yt.mean()), 'decision_threshold': 0.5}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).to_csv(out / 'experiments.csv', index=False)
    save_json(out / 'metrics.json', metrics)
    save_json(out / 'best_parameters.json', search.best_estimator_.get_params(deep=True))
    save_json(out / 'run_metadata.json', {'configuration': cfg, 'splits': metadata,
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ['pandas', 'numpy', 'scikit-learn', 'joblib']},
        'selection_metric': 'average_precision', 'folds': 3, 'seed': 42,
        'candidate_count': len(search.cv_results_['params'])})
    joblib.dump(search.best_estimator_, out / 'best_model.joblib')
    print(json.dumps(metrics, indent=2))
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['validate', 'prepare', 'train', 'all'])
    p.add_argument('--config', default='config.json')
    p.add_argument('--splits', default='artifacts/splits')
    p.add_argument('--output', default='artifacts/model')
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.stage == 'validate':
        _, summary = validate(cfg)
        print(json.dumps(summary, indent=2, default=str))
    if args.stage in ['prepare', 'all']:
        prepare(cfg, Path(args.splits))
    if args.stage in ['train', 'all']:
        train_model(cfg, Path(args.splits), Path(args.output))

if __name__ == '__main__':
    main()
