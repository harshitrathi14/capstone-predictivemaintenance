# Predictive Maintenance — Interim Project

Analysis of the supplied `engine_data.csv`: 19,535 observations, six numeric sensor features, and the binary `Engine Condition` target. The original upload remains in the root; the analysis copy is in `data/`. `config.json` contains the schema, source note and report name (Harshit Rathi).

**Interpretation:** The uploaded CSV is a byte-for-byte match to Parv Modi’s Kaggle dataset (version 1). Its source metadata does not define the class meanings, and external research gives conflicting mappings. Class 0/class 1 terminology is retained throughout; class 1 is positive only for statistical evaluation. Source links and the conflicting evidence are documented in the PDF and HTML. Without timestamps or engine identifiers, this analysis evaluates condition classification, not advance failure prediction.

## Submission files

- `reports/Harshit_Rathi_PredictiveMaintenance_InterimReport.pdf`
- `reports/Harshit_Rathi_PredictiveMaintenance_Notebook.html`

The executed source notebook is `notebooks/Harshit_Rathi_PredictiveMaintenance.ipynb`. Its analytical code is embedded so the HTML contains the complete implementation and outputs. Dataset provenance has been verified. The healthy/faulty mapping still requires confirmation from the course data dictionary or dataset creator.

## Reproduce

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/deliverables.py
```

This runs the complete notebook, including validation, training-only EDA, splitting, 16-candidate cross-validation, held-out evaluation and plotting, then exports HTML and renders a separate PDF. Run from the project root. The report name and label documentation are configured in `config.json`.

The existing configured `.venv` can be used directly:

```sh
.venv/bin/python src/deliverables.py
.venv/bin/python -m unittest discover -s tests -v
```

Individual workflow stages:

```sh
python src/pipeline.py validate
python src/pipeline.py prepare
python src/pipeline.py train
```

To change only PDF layout without refitting models, use `python src/deliverables.py --report-only`. This requires the executed notebook and existing analysis artifacts with matching configuration. The engine-specific EDA/report generator expects the supplied six-sensor schema; `pipeline.py` itself supports configurable binary-classification schemas.

## Evidence and rubric coverage

| Requirement | Implementation / output |
|---|---|
| Registration | Exact schema and numeric/target checks; source SHA-256, missing and duplicate summaries |
| EDA | Training summaries, univariate distributions, class comparisons, multivariate correlations and data-specific business interpretations |
| Preparation | No missing values/duplicates observed; all six sensors retained; stratified 80/20 CSV splits; preprocessing fitted inside CV |
| Artifact handoff | Separate GitHub Actions preparation/training jobs using upload/download-artifact; split checksum validation |
| Experiment tracking | 16 parameter combinations × 3 folds; all parameters, scores, ranks and timing in `artifacts/model/experiments.csv` |
| Best model | Preprocessing and fitted random forest in `artifacts/model/best_model.joblib` |
| Evaluation | Held-out metrics, prior baseline, confusion matrix and ROC/precision-recall plots; no test-set tuning |
| Submission | Numbered PDF pages, figures/tables, methods, findings, recommendations and raw-code appendix; executed HTML notebook |
| Reproducibility | `artifacts/model/run_metadata.json` records environment, configuration, seed and source/split hashes |

The GitHub workflow is provided locally; this folder is not a Git repository and no hosted workflow run or remote commit has been performed. The workflow uploads the best model and logs as an artifact available for a later repository commit. Add the data, source, configuration and workflow to the intended repository before running it there.

Tests use temporary synthetic fixtures only for software verification, covering schema/label rejection, artifact tampering, disjoint splits, model serialization and unseen categories. Submission results use the supplied engine dataset.
