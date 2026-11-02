# Dataset registration

`engine_data.csv` is a byte-for-byte copy of the user's root-level upload, supplied on 2 November 2026. It contains 19,535 rows and seven columns: Engine rpm, Lub oil pressure, Fuel pressure, Coolant pressure, lub oil temp, Coolant temp, and Engine Condition.

The file was verified byte-for-byte against [Parv Modi’s Kaggle dataset](https://www.kaggle.com/datasets/parvmodi/automotive-vehicles-engine-health-dataset), version 1. Its source metadata does not define the target labels; external research conflicts on their meaning. Acquisition process and pressure/temperature units remain unverified. Do not infer fault/health semantics from the numeric labels. Class 1 is used as the positive evaluation label only. The original root upload is preserved.

SHA-256: `616c7b985ec00e299061f02b11fd866f8eca16b5e8d15d4a61a5fbfd116b8045`

Validation: `python src/pipeline.py validate`. Configuration: `config.json`.
