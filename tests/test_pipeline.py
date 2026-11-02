"""Synthetic fixtures verify code only; they are never assignment evidence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pipeline

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_root = pipeline.ROOT
        pipeline.ROOT = Path(self.tmp.name)
        (pipeline.ROOT / 'data').mkdir()
        rng = np.random.default_rng(42)
        self.df = pd.DataFrame({'id':range(90), 'sensor':rng.normal(size=90),
            'type':['A','B','C']*30, 'failure':[0,0,1]*30})
        self.df.to_csv(pipeline.ROOT/'data/fixture.csv', index=False)
        self.cfg = {'dataset':'data/fixture.csv','source':'Synthetic test fixture only',
            'target':'failure','positive_label':1,'expected_columns':list(self.df),
            'numeric_columns':['sensor'],'drop_columns':['id']}
    def tearDown(self):
        pipeline.ROOT = self.old_root
        self.tmp.cleanup()
    def test_schema_rejected(self):
        self.cfg['expected_columns'].append('missing_sensor')
        with self.assertRaisesRegex(ValueError,'Schema mismatch'):
            pipeline.validate(self.cfg)
    def test_unknown_positive_rejected(self):
        self.cfg['positive_label'] = 2
        with self.assertRaisesRegex(ValueError,'positive_label'):
            pipeline.validate(self.cfg)
    def test_modified_split_rejected(self):
        split = pipeline.ROOT/'splits'
        pipeline.prepare(self.cfg, split)
        with (split/'train.csv').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            pipeline.train_model(self.cfg, split, pipeline.ROOT/'model')
    def test_split_is_disjoint_and_training_roundtrips(self):
        split = pipeline.ROOT/'splits'; out = pipeline.ROOT/'model'
        train, test = pipeline.prepare(self.cfg, split)
        self.assertNotIn('id', train)
        self.assertFalse(set(train.sensor) & set(test.sensor))
        metrics = pipeline.train_model(self.cfg, split, out)
        self.assertEqual(len(pd.read_csv(out/'experiments.csv')),16)
        self.assertEqual(sum(map(sum,metrics['random_forest']['confusion_matrix'])),len(test))
        model = pipeline.joblib.load(out/'best_model.joblib')
        self.assertEqual(len(model.predict(test.drop(columns='failure'))),len(test))
        unseen = test.drop(columns='failure').copy(); unseen['type'] = 'UNSEEN'
        self.assertEqual(len(model.predict(unseen)),len(test))

if __name__ == '__main__':
    unittest.main()
