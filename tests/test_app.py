"""Verify the deployed-model contract and actual Streamlit prediction interaction."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import load_bundle, predict_observation
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.profile, _ = load_bundle()

    def readings(self):
        return {name: p['median'] for name, p in self.profile['summary'].items()}

    def test_saved_model_prediction_matches_form_dataframe(self):
        frame, predicted, score, outside = predict_observation(self.readings(), self.model, self.profile)
        self.assertEqual(list(frame), list(self.model.feature_names_in_))
        self.assertEqual(predicted, int(self.model.predict(frame)[0]))
        self.assertTrue(0 <= score <= 1)
        self.assertFalse(outside)

    def test_incomplete_and_nonfinite_readings_rejected(self):
        values = self.readings()
        values.pop(next(iter(values)))
        with self.assertRaisesRegex(ValueError, 'six required'):
            predict_observation(values, self.model, self.profile)
        values = self.readings()
        values['Engine rpm'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            predict_observation(values, self.model, self.profile)

    def test_app_submits_and_warns_on_outside_range(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py')).run()
        self.assertFalse(app.exception)
        app.number_input(key='Engine rpm').set_value(100000.0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 2)
        self.assertIn('Outside the observed training range', app.warning[0].value)
        self.assertEqual(float(app.dataframe[0].value['Engine rpm'].iloc[0]), 100000.0)


if __name__ == '__main__':
    unittest.main()
