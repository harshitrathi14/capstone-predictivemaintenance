"""Load the repository model and validate a single engine observation."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_bundle(model_dir=None):
    directory = Path(model_dir) if model_dir is not None else ROOT / 'models'
    metadata = json.loads((directory / 'run_metadata.json').read_text())
    profile = json.loads((directory / 'input_profile.json').read_text())
    model_path = directory / 'best_model.joblib'
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != profile['model_sha256']:
        raise ValueError('Model and input profile do not match. Refresh the model bundle.')
    for package in ['scikit-learn', 'numpy', 'pandas', 'joblib']:
        if version(package) != metadata['packages'][package]:
            raise ValueError(f'{package} differs from the training environment. Install requirements.txt.')
    # Only the trusted model committed by this project is loaded, never user uploads.
    model = joblib.load(model_path)
    if list(model.feature_names_in_) != profile['features']:
        raise ValueError('Model feature order does not match the input profile.')
    return model, profile, metadata


def predict_observation(values, model, profile):
    features = profile['features']
    if set(values) != set(features):
        raise ValueError('Provide exactly the six required sensor readings.')
    frame = pd.DataFrame([values], columns=features).astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Every sensor reading must be a finite number.')
    outside = [name for name in features
               if not profile['summary'][name]['minimum'] <= frame.iloc[0][name]
               <= profile['summary'][name]['maximum']]
    prediction = int(model.predict(frame)[0])
    positive_index = list(model.classes_).index(1)
    score = float(model.predict_proba(frame)[0, positive_index])
    return frame, prediction, score, outside
