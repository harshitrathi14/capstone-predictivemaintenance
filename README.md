# Predictive Maintenance — Final Project

Harshit Rathi · AIML Capstone Project 3

Engine-condition classification using six sensor readings, a tuned random forest, a Streamlit Community Cloud app and an automated GitHub Actions workflow.

## Final submission

The two submission files are generated in `submission/final/`:

- `Harshit_Rathi_PredictiveMaintenance_FinalReport.pdf`
- `Harshit_Rathi_PredictiveMaintenance_Notebook.html`

The executed source notebook is `notebooks/Harshit_Rathi_PredictiveMaintenance_Final.ipynb`. The original interim submission remains in `submission/`.

## Interpretation and source

The supplied dataset has 19,535 observations, six numeric sensors and the binary `Engine Condition` target. The source CSV is in `data/engine_data.csv`. The original upload was verified against [Parv Modi’s Kaggle dataset](https://www.kaggle.com/datasets/parvmodi/automotive-vehicles-engine-health-dataset). Source notes and conflicting external label definitions are recorded in `config.json`.

The healthy/faulty label mapping and pressure/temperature units remain unverified. The app retains the original class 0/class 1 labels. Without timestamps, engine identifiers and future outcomes, this is condition classification, not demonstrated advance failure prediction. No maintenance savings are claimed.

## Streamlit Community Cloud

Deploy with these settings:

| Setting | Value |
|---|---|
| Repository | `harshitrathi14/capstone-predictivemaintenance` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| Python version (Advanced settings) | `3.12` |
| Dependencies | `requirements.txt` |
| Secrets | None required |

The app loads `models/best_model.joblib`, collects six readings in a dataframe, validates inputs and displays the predicted dataset class and class-1 score. The score is not a calibrated failure probability. Input defaults and observed ranges come only from the training split. The saved model checksum and inference package versions are checked before loading.

## Reproduce

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Individual training stages:

```sh
python src/pipeline.py validate
python src/pipeline.py prepare
python src/pipeline.py train
python -m unittest discover -s tests -v
```

The model test fixtures are synthetic and used only for software verification. Report results use the supplied engine dataset.

## Automation

`.github/workflows/pipeline.yml` runs on relevant pushes to `main` and manual dispatch:

1. **prepare:** validate data, test the pipeline, save stratified train/test CSVs and upload the split artifact.
2. **train:** download and verify the split artifact, tune 16 candidates with three-fold cross-validation, evaluate the holdout, save the model and logs, and test the app against the new model.
3. **persist-model:** download the model artifact and automatically commit changed model files and records to `main`.

Model-only commits do not retrigger training. Core model dependencies are pinned. After initial deployment, Community Cloud follows the selected branch. The workflow run that produced the model is recorded in `models/workflow_run.json`.

## Report generation and evidence

```sh
python src/final_deliverables.py
```

The generator requires `evidence/final_evidence.json` and actual repository, workflow and deployed-app screenshots. It executes the full final notebook, verifies the committed model's recorded metrics and split hashes, exports the HTML, renders the PDF, and copies the two files into `submission/final/`. The report uses the committed hosted model; the notebook distinguishes its results from a fresh local reproduction. Both are fully logged.

To adjust report layout without refitting, use `python src/final_deliverables.py --report-only` after an executed final notebook exists. Model checks prevent accidentally pairing a report with a different model. Generation timestamps and file checksums are saved locally in `reports/final/submission_manifest.json`; this is not a portal submission receipt.
