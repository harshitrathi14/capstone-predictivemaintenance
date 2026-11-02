"""Execute a self-contained analysis notebook and render its evidence as a business PDF."""
import argparse
import json
import re
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape
import nbformat
import pandas as pd
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                               TableStyle, PageBreak, Preformatted)
from pipeline import ROOT, load_config


def filenames(cfg):
    prefix = re.sub(r'[^\w-]+', '_', cfg['learner_name']) + '_PredictiveMaintenance'
    return (ROOT/'notebooks'/(prefix+'.ipynb'), ROOT/'reports'/(prefix+'_Notebook.html'),
            ROOT/'reports'/(prefix+'_InterimReport.pdf'))


def execute_notebook(cfg):
    notebook_path, html_path, _ = filenames(cfg)
    notebook_path.parent.mkdir(exist_ok=True)
    html_path.parent.mkdir(exist_ok=True)
    # Embed the analytical source so the exported notebook contains all model/EDA code.
    pipeline_code = (ROOT/'src/pipeline.py').read_text().split('\ndef main():')[0]
    pipeline_code = pipeline_code.replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path.cwd()')
    analysis_code = (ROOT/'src/analysis.py').read_text()
    evidence_text = '\n\n'.join(
        f"- [{item['title']}]({item['url']}): {item['finding']}"
        for item in cfg.get('label_evidence', []))
    cells = [
        nbformat.v4.new_markdown_cell('# Predictive Maintenance — Interim Analysis\n\n'
            + cfg['learner_name'] + '\n\nSource: ' + cfg['source'] + '\n\n' + cfg['label_note']
            + '\n\nReproduction: run from the project root with data/engine_data.csv present; install requirements.txt. '
              'The analytical source is embedded below. All EDA uses training rows; the test set is reserved for final evaluation.'
            + '\n\n### Source verification and conflicting label definitions\n\n' + evidence_text
            + '\n\nThe disagreement is unresolved. Sensor averages cannot establish label semantics without verified ground truth. No labels have been reversed or renamed.'),
        nbformat.v4.new_code_cell(pipeline_code + '\nprint("Validation and modeling functions loaded; random seed = 42.")'),
        nbformat.v4.new_markdown_cell('## 1. Data registration and preparation\n\n'
            'Methodology: validate all seven columns, numeric features, binary labels, missing values and duplicates; '
            'hash the input. Retain all six sensors, remove exact duplicates if present and stratify an 80/20 split. '
            'Learn imputation inside cross-validation. No timestamp or equipment identifier is available for a time/group split.'),
        nbformat.v4.new_code_cell('cfg = ' + repr(cfg) + '''
splits = Path('artifacts/splits')
train, test = prepare(cfg, splits)
print('Label interpretation:', cfg['label_note'])
print('Training shape:', train.shape, 'Test shape:', test.shape)
print('No features dropped; no missing values or exact duplicates in the supplied file.')
'''),
        nbformat.v4.new_markdown_cell('## 2. Exploratory analysis\n\n'
            'Methodology: summarize training distributions, compare feature means and boxplots by class, '
            'calculate Pearson correlations, and flag 1.5-IQR extremes without deleting them. '
            'Sensor units, collection process and target meanings were not provided.'),
        nbformat.v4.new_code_cell(analysis_code + '\nnotes = explore(train, cfg, Path("reports"))'),
        nbformat.v4.new_markdown_cell('## 3. Model building and experimentation tracking\n\n'
            'Methodology: evaluate 16 random forest candidates using three shuffled stratified folds, seed 42. '
            'Rank by class-1 average precision; refit the winner on the training split. The test set is not used for tuning. '
            'A prior-probability dummy model is the baseline. The default decision threshold is 0.5.'),
        nbformat.v4.new_code_cell('''metrics = train_model(cfg, splits, Path('artifacts/model'))
print('Table 4. Full experiment log, including all fold scores and tuned parameters.')
experiments = pd.read_csv('artifacts/model/experiments.csv')
with pd.option_context('display.max_columns', None, 'display.max_rows', None):
    display(experiments)
diagnostics(test, cfg, Path('artifacts/model'), Path('reports'))
print('Environment and dataset lineage:')
print(Path('artifacts/model/run_metadata.json').read_text())
'''),
        nbformat.v4.new_markdown_cell('## 4. Business interpretation and limitations\n\n'
            'This is condition classification, not demonstrated prediction of future breakdown. '
            'Confirm which label denotes the maintenance-relevant condition before interpreting errors as missed faults or false alarms. '
            'Use validation data and agreed intervention costs to choose a threshold; do not optimize on the test set. '
            'Validate on new equipment and later time periods before deployment. No financial savings are claimed.')
    ]
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}})
    ExecutePreprocessor(timeout=1800, kernel_name='python3').preprocess(nb, {'metadata': {'path':str(ROOT)}})
    nbformat.write(nb, notebook_path)
    html, _ = HTMLExporter().from_notebook_node(nb)
    html_path.write_text(html)
    return nb


def render_report(cfg, nb):
    _, _, pdf_path = filenames(cfg)
    out = ROOT/'reports'
    metrics = json.loads((ROOT/'artifacts/model/metrics.json').read_text())
    registration = json.loads((ROOT/'artifacts/splits/registration.json').read_text())
    split = json.loads((ROOT/'artifacts/splits/split_metadata.json').read_text())
    run = json.loads((ROOT/'artifacts/model/run_metadata.json').read_text())
    if run['configuration'] != cfg:
        raise ValueError('Configuration differs from executed evidence; rerun the notebook.')
    notes = json.loads((out/'eda_insights.json').read_text())
    stats = pd.read_csv(out/'feature_summary.csv', index_col=0)
    comparison = pd.read_csv(out/'class_comparison.csv', index_col=0)
    log = pd.read_csv(ROOT/'artifacts/model/experiments.csv')
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceAfter = 7
    styles['Heading1'].textColor = colors.HexColor('#17354d')
    styles['Heading2'].textColor = colors.HexColor('#176d70')
    styles.add(ParagraphStyle('Caption', parent=styles['BodyText'], fontSize=9, leading=12))
    styles.add(ParagraphStyle('Cover', parent=styles['Title'], fontSize=25, leading=30,
                              textColor=colors.HexColor('#17354d'), alignment=TA_CENTER))
    styles.add(ParagraphStyle('SmallCode', parent=styles['Code'], fontSize=6.8, leading=8.2))
    story = []

    def p(text, style='BodyText'):
        story.append(Paragraph(escape(str(text)), styles[style]))

    def heading(text):
        p(text, 'Heading1')

    def table(title, headers, rows, widths=None):
        p(title, 'Heading2')
        cellstyle = ParagraphStyle('cell', parent=styles['BodyText'], fontSize=8, leading=10, spaceAfter=0)
        data = [[Paragraph(escape(str(item)), cellstyle) for item in row] for row in [headers]+rows]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4eeee')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#176d70')),
                              ('LINEBELOW',(0,1),(-1,-1),.25,colors.lightgrey),
                              ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                              ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([t, Spacer(1,10)])

    def figure(number, caption, max_height=330):
        image = Image(str(out/f'figure_{number}.png'))
        scale = min(475/image.imageWidth, max_height/image.imageHeight)
        image.drawWidth = image.imageWidth*scale
        image.drawHeight = image.imageHeight*scale
        story.extend([image, Spacer(1,5)])
        p(caption, 'Caption')

    def page():
        story.append(PageBreak())

    p('Predictive Maintenance', 'Cover')
    p('Interim Project Report', 'Title')
    p(cfg['learner_name'], 'Title')
    p('Engine condition classification | 2 November 2026', 'Title')
    heading('Executive overview')
    p(f"The supplied dataset contains {registration['rows']:,} observations and six sensor measurements. "
      'The analysis tests whether these measurements distinguish the two Engine Condition classes. '
      'A reproducible workflow validates the data, creates saved splits, tunes a random forest, and compares it with a majority-class baseline.')
    p(f"On {split['test_rows']:,} held-out observations, the random forest achieved "
      f"{metrics['random_forest']['balanced_accuracy']:.2f} balanced accuracy and "
      f"{metrics['random_forest']['average_precision']:.2f} average precision for class 1, versus "
      f"{metrics['baseline']['balanced_accuracy']:.2f} and {metrics['baseline']['average_precision']:.2f} for the baseline.")
    p(cfg['label_note'])
    p('The result is evidence of predictive association within this file. There is no time horizon, '
      'equipment identifier or prospective failure history, so advance warning of a breakdown and production readiness have not been established.')
    heading('Business objective')
    p('Support a maintenance analyst in reviewing engine-condition observations using rpm, oil/fuel/coolant pressures and oil/coolant temperatures. '
      'A successful operational system must reduce costly missed conditions while limiting unnecessary inspections. '
      'The current interim evaluation measures classification quality; a cost-benefit claim requires label definitions and intervention costs.')
    page()

    heading('1. Data registration and background')
    p('Methodology', 'Heading2')
    p('Load the CSV from the repository data folder; verify the exact seven-column schema, numeric sensor values, '
      'binary target, missing values, duplicates and finite values; record SHA-256 for reproducibility.')
    p('Source: '+cfg['source'])
    p('The healthy/faulty mapping remains unverified because external sources conflict. '
      'Nafar describes class 1 as healthy, while Chen describes class 1 as requiring maintenance. '
      'The original source metadata does not resolve this disagreement. See Appendix A for references. '
      'No class has been relabeled or interpreted as better/worse condition.')
    p('Collection background: the supplied file is a tabular snapshot. No documentation establishes where or when engines were measured, '
      'how the condition label was assigned, whether rows belong to repeated engines, or whether the observations were simulated. '
      'These gaps limit generalization and physical interpretation; no provenance is inferred from the filename.')
    counts = registration['class_counts']
    table('Table 1. Registered dataset overview', ['Check', 'Result'], [
        ['Rows / columns', f"{registration['rows']:,} / {len(registration['columns'])}"],
        ['Predictors / outcome', 'Six numeric sensors / Engine Condition (0 or 1)'],
        ['Class 0',f"{counts['0']:,} ({100*counts['0']/registration['rows']:.2f}%)"],
        ['Class 1',f"{counts['1']:,} ({100*counts['1']/registration['rows']:.2f}%)"],
        ['Missing values / exact duplicate rows', f"{sum(registration['missing'].values())} / {registration['duplicate_rows']}"],
        ['Unnecessary / leakage fields', 'No identifier or outcome-derived field is evident; all six sensors retained'],
        ['Units and label definitions', 'rpm is explicit; pressure/temperature units and label meanings unconfirmed']], [165,310])
    p('Schema: '+', '.join(registration['columns'])+'.')
    p('The raw upload is preserved as engine_data.csv; the validated analysis copy is data/engine_data.csv. '
      'Their checksum is recorded in Appendix A. No external images or replacement datasets were used.')
    page()

    heading('2. Exploratory data analysis')
    p('Methodology', 'Heading2')
    p('Conduct exploratory data analysis (EDA) only on the 15,628 training observations. Use distribution summaries and histograms for univariate analysis; '
      'class-conditioned means and boxplots for bivariate analysis; a correlation matrix for multivariate analysis. '
      'The 3,907 test observations are excluded from exploratory decisions.')
    table('Table 2. Training sensor summary', ['Sensor','Mean','Std. dev.','Median','Minimum','Maximum'],
          [[name]+[f'{row[k]:.2f}' for k in ['mean','std','50%','min','max']] for name,row in stats.iterrows()], [120,71,71,71,71,71])
    p('Values shown as 0.00 are rounded small positive pressures, not zero-valued or missing readings. Units other than rpm were not supplied.')
    figure(1, 'Figure 1. Class 1 is the more frequent class; the imbalance is moderate, not an extremely rare-event setting.', 220)
    p(notes[0])
    page()

    heading('2.1 Univariate patterns and unusual readings')
    p('Outlier checks use the interquartile range (IQR), the difference between the 75th and 25th percentiles. Values beyond 1.5 IQR below/above those percentiles are flagged for review.')
    figure(2, 'Figure 2. Histograms use the entire training range so unusually high sensor values remain visible.', 315)
    p('Engine rpm and the pressure features have upper tails; oil temperature is concentrated near the mid-to-high 70s with a smaller high-temperature tail. '
      'A wide coolant-temperature range is driven partly by a small set of high observations, so typical operation is better described by both median and spread.')
    p(notes[4])
    table('Table 3. Training outlier checks', ['Sensor', 'Outside 1.5-IQR limits', 'Handling'],
          [[name,str(int(row['IQR flags'])),'Retained; no verified physical limit supplied'] for name,row in stats.iterrows()], [130,120,225])
    p('Business implication: review extreme readings with sensor/domain experts. Automatically deleting them could remove the very operating states a condition model should recognize.')
    page()

    heading('2.2 Bivariate class comparisons')
    table('Table 4. Mean sensor readings by class (training set)', ['Sensor','Class 0','Class 1','Difference (1 - 0)'],
          [[name]+[f'{v:.2f}' for v in row] for name,row in comparison.iterrows()], [160,95,95,125])
    figure(3, 'Figure 3. Each panel compares one sensor with Engine Condition; overlap limits single-sensor rules.', 300)
    p(notes[1])
    p('Class 1 also has higher mean fuel pressure and lower mean oil temperature in this sample. '
      'These differences support testing a combined sensor model, but their direction cannot be translated into “better” or “worse” condition until the label definitions are verified.')
    page()

    heading('2.3 Multivariate associations and implications')
    figure(4, 'Figure 4. Pearson correlations summarize linear relationships; encoded target correlation is interpreted as a class association.', 350)
    p(notes[2])
    p(notes[3])
    p('Business implication: there is no strong linear redundancy that justifies dropping a sensor on correlation grounds alone. '
      'A tree ensemble can test nonlinear combinations while retaining the available measurement context. Correlation does not demonstrate causation.')
    p(notes[5])
    page()

    heading('3. Data preparation')
    p('Methodology', 'Heading2')
    p('Retain the six numeric predictors, check for duplicate predictor rows with conflicting labels, '
      'remove exact duplicates if present and stratify an 80/20 split using seed 42. '
      'The observed dataset requires no row deletion or missing-value replacement. Median imputation remains in the fitted pipeline as a reproducible safeguard, '
      'and is learned independently within every training fold.')
    table('Table 5. Preparation decisions and rationale', ['Step','Decision and reason'], [
        ['Column removal','None. There are no IDs or explicit post-outcome columns in the supplied schema. Prediction-time sensor availability still needs confirmation.'],
        ['Duplicate / missing handling','No missing values or exact duplicates observed. No rows removed. Conflicting identical predictor rows are rejected.'],
        ['Extreme values','Retained. No documented physical ranges justify deletion or capping.'],
        ['Scaling / encoding','No scaling needed for the tree model; all observed predictors are numeric. The reusable pipeline also supports categorical encoding.'],
        ['Training / test rows',f"{split['train_rows']:,} / {split['test_rows']:,}; stratified, random_state = 42."],
        ['Feature engineering','Use the six original readings for an interpretable baseline. No unsupported unit-dependent ratios or time features are invented.'],
        ['Leakage prevention','No target used as a predictor; imputation fit within folds; hyperparameter selection excludes test data.']], [135,340])
    p('Saved artifacts: artifacts/splits/train.csv, test.csv, registration.json and split_metadata.json. '
      'The metadata includes source/split hashes and the class mapping. The training job validates the hashes before loading the split artifacts.')
    p('Workflow design: the GitHub Actions prepare job uploads train-test-splits; the dependent train job downloads those files, '
      'fits the models and uploads model-and-experiments. The local artifact handoff was executed; a hosted GitHub run has not been performed because this directory is not linked to a repository.')
    p('Split limitation: a random split assumes sufficiently independent rows. The CSV does not allow equipment-group or chronological validation. '
      'If repeated-engine data or timestamps become available, reevaluate with group/time separation before operational use.')
    page()

    heading('4. Model building and experiment tracking')
    p('Methodology', 'Heading2')
    p('Compare a prior-probability dummy baseline with a random forest. '
      'Tune 16 combinations using three shuffled, stratified cross-validation (CV) folds (48 fits), selecting the highest mean validation average precision (AP) for class 1. '
      'Refit the selected pipeline on the complete training split, then evaluate on the held-out test set at the default threshold of 0.5.')
    table('Table 6. Hyperparameter search space', ['Parameter','Values'], [
        ['n_estimators','100, 200'],['max_depth','Unrestricted, 10'],['min_samples_leaf','1, 3'],
        ['class_weight','None, balanced'],['Fixed settings','random_state = 42; model n_jobs = 1; grid n_jobs = 4']], [180,295])
    p('Average precision measures ranking of class 1 across thresholds. Its no-skill reference is the class-1 prevalence. '
      'Balanced accuracy averages recall for both classes, so a majority-only classifier scores 0.50. '
      'Area under the receiver operating characteristic curve (ROC AUC) measures ranking discrimination. '
      'Precision is the fraction of predicted class-1 rows actually in class 1; recall is the fraction of actual class-1 rows found. '
      'F1 is their harmonic mean. These last three metrics describe the selected 0.5 operating point.')
    best = log.loc[log['rank_test_score'].idxmin()]
    params = metrics['best_parameters']
    p(f"Selected model: {params['model__n_estimators']} trees, maximum depth {params['model__max_depth']}, "
      f"minimum {params['model__min_samples_leaf']} observation(s) per leaf, and {params['model__class_weight']} class weighting. "
      'Best mean cross-validation average precision: '
      f"{metrics['best_cv_average_precision']:.2f} (fold standard deviation {best['std_test_score']:.2f}). "
      f"The corresponding mean training average precision is {best['mean_train_score']:.2f}; the training/validation gap is a check for overfitting.")
    ordered = log.sort_values(['rank_test_score','mean_fit_time']).head(5)
    table('Table 7. Top five candidates by mean validation average precision', ['Trees','Depth','Leaf min.','Weights','CV AP','Train AP'],
          [[int(r['param_model__n_estimators']), 'None' if pd.isna(r['param_model__max_depth']) else int(r['param_model__max_depth']),
            int(r['param_model__min_samples_leaf']), 'None' if pd.isna(r['param_model__class_weight']) else r['param_model__class_weight'],
            f"{r['mean_test_score']:.2f}",f"{r['mean_train_score']:.2f}"] for _,r in ordered.iterrows()], [60,65,65,95,95,95])
    p('All candidate parameters, fold scores, training scores, ranks and timing are logged in experiments.csv and displayed in the HTML notebook. '
      'The saved best_model.joblib contains preprocessing and the fitted forest. best_parameters.json and run_metadata.json record full settings, environment versions and input lineage. '
      'The model is available for a later repository commit; no remote commit was made.')
    page()

    heading('4.1 Held-out evaluation')
    table('Table 8. Held-out performance (positive class = 1)', ['Metric','Prior baseline','Random forest'],
          [[key.replace('_',' ').title(),f"{metrics['baseline'][key]:.2f}",f"{metrics['random_forest'][key]:.2f}"]
           for key in ['average_precision','roc_auc','balanced_accuracy','precision','recall','f1']], [235,120,120])
    figure(6, 'Figure 6. The diagonal ROC line and prevalence-level precision line are reference baselines. Curves describe the held-out sample only.', 220)
    p(f"The test set contains {100*metrics['test_prevalence']:.2f}% class-1 observations. "
      f"Average precision improves by {metrics['random_forest']['average_precision']-metrics['baseline']['average_precision']:.2f} "
      'over the prior baseline. This indicates additional ranking information beyond the class proportions, while remaining specific to the supplied sample.')
    p('The baseline predicts class 1 for every row because class 1 is the majority. Its perfect class-1 recall is therefore not evidence of useful separation; it recognizes none of class 0.')
    p(f"At threshold 0.5, the forest's class-1 F1 is {metrics['random_forest']['f1']:.2f}, "
      f"below the baseline's {metrics['baseline']['f1']:.2f}. Selection optimized average precision, not F1. "
      'The forest exchanges some class-1 recall for class-0 recognition. Better ranking does not guarantee a better operating point for every metric or business cost.')
    page()

    heading('4.2 Errors and operational meaning')
    figure(5, 'Figure 5. Rows are actual classes and columns are predicted classes; class semantics are unconfirmed.', 285)
    tn, fp = metrics['random_forest']['confusion_matrix'][0]
    fn, tp = metrics['random_forest']['confusion_matrix'][1]
    p(f'Of {tn+fp:,} actual class-0 observations, {tn:,} were classified as 0 and {fp:,} as 1. '
      f'Of {fn+tp:,} actual class-1 observations, {tp:,} were classified as 1 and {fn:,} as 0. '
      f'Class-0 recall is {tn/(tn+fp):.2f}; class-1 recall is {tp/(tp+fn):.2f}.')
    p(f'The model predicts class 1 for {tp+fp:,} of {split["test_rows"]:,} observations '
      f'({100*(tp+fp)/split["test_rows"]:.2f}%). These predictions are not maintenance alerts. '
      'Neither error type is called a missed fault or false alarm until the healthy/faulty mapping is verified.')
    heading('5. Actionable insights and next steps')
    p('1. Obtain the data dictionary first: confirm label meanings, sensor units, acquisition context and prediction-time availability. '
      'Until then, use the scores for analytical review rather than automatic maintenance orders.')
    p('2. Review extreme coolant readings and the observed rpm/class relationship with an engine specialist. '
      'Retain suspicious readings in the analysis until measurement errors can be distinguished from unusual operating states.')
    p('3. Once the intervention class is confirmed, choose a threshold on validation data using agreed costs of missed conditions and unnecessary inspections. '
      'The current threshold was not cost-optimized, and these observations do not establish financial savings.')
    p('4. Collect engine identifiers, timestamps and future failure outcomes. Evaluate on unseen engines and later periods, '
      'estimate uncertainty, and run a supervised pilot before introducing operational alerts. Deployment and monitoring are outside this interim scope.')
    page()

    heading('Appendix A. Registration and reproducibility evidence')
    p('Dataset SHA-256: '+registration['sha256'])
    p('Source: '+cfg['source'])
    p('Source verification and conflicting label definitions', 'Heading2')
    for item in cfg.get('label_evidence', []):
        p(item['title'] + '. ' + item['finding'])
        p(item['url'], 'Caption')
    p('The source download and uploaded CSV have identical SHA-256 hashes. '
      'The cited disagreement prevents a verified healthy/faulty interpretation; class 0/class 1 labels are retained throughout.')
    p('Code, analysis and plots were generated for this project. Software libraries used: pandas, NumPy, scikit-learn, '
      'Matplotlib, Seaborn, joblib, Jupyter nbformat/nbconvert and ReportLab. No external visual assets were used.')
    for name,digest in split['split_sha256'].items():
        p(name+' CSV SHA-256: '+digest)
    table('Table A1. Execution environment', ['Component','Version / setting'],
          [['Python',run['python']]]+[[k,v] for k,v in run['packages'].items()]+[['Seed',42],['Candidate count',run['candidate_count']],['CV folds',run['folds']]], [200,275])
    p('Verification evidence: the notebook was executed end to end and exported after execution. '
      'The HTML includes full experiment output; reports/ contains the numbered plots and EDA tables. '
      'Tests cover invalid schema/labels, disjoint splits, model serialization and unseen-category handling.')
    page()
    heading('Appendix B. Extended experiment log')
    p('Table B1 below shows every candidate at full analytical precision for audit (main-body values use two decimal places). '
      'Full fold scores, parameters and timing appear in the HTML notebook and artifacts/model/experiments.csv.')
    table('Table B1. All 16 candidate results', ['Trees','Depth','Leaf','Weight','CV AP','CV std.','Rank'],
          [[int(r['param_model__n_estimators']), 'None' if pd.isna(r['param_model__max_depth']) else int(r['param_model__max_depth']),
            int(r['param_model__min_samples_leaf']), 'None' if pd.isna(r['param_model__class_weight']) else r['param_model__class_weight'],
            f"{r['mean_test_score']:.5f}",f"{r['std_test_score']:.5f}",int(r['rank_test_score'])] for _,r in log.iterrows()], [50,50,45,90,90,90,60])
    page()
    heading('Appendix B (continued). Fold scores and execution timing')
    p('Candidate IDs follow the row order in Table B1 and the exported experiment log. '
      'AP is average precision. Time is the mean fit time in seconds across the three folds. '
      'Detailed training-fold scores and scoring times remain available in the full HTML log.')
    table('Table B2. Validation-fold scores and mean fit times for all candidates',
          ['Candidate','Fold 1 AP','Fold 2 AP','Fold 3 AP','Train AP','Fit seconds'],
          [[i+1]+[f'{r[k]:.5f}' for k in ['split0_test_score','split1_test_score','split2_test_score','mean_train_score']]
           +[f"{r['mean_fit_time']:.2f}"] for i,r in log.iterrows()], [70,81,81,81,81,81])
    page()
    heading('Appendix C. Executed analytical code')
    p('The following code cells are the exact analytical program contained in the executed notebook, with long lines wrapped for print. '
      'Use the .ipynb file or HTML for the original formatting. The report layout generator is available separately in src/deliverables.py.')
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        p(f'Code cell {cell.execution_count}', 'Heading2')
        lines = []
        for line in cell.source.splitlines():
            lines.extend(textwrap.wrap(line, width=105, replace_whitespace=False, drop_whitespace=False) or [''])
        story.append(Preformatted('\n'.join(lines), styles['SmallCode']))

    page()
    heading('Appendix D. Workflow artifact handoff configuration')
    p('The following GitHub Actions workflow uploads the saved training/test split artifact from the preparation job '
      'and downloads it in the dependent training job. The fitted model and experiment logs are then uploaded as a second artifact. '
      'This is configuration evidence, not evidence of a completed hosted run. Local CSV preparation and loading were executed; '
      'a GitHub run and remote model commit remain outstanding.')
    p('Source: .github/workflows/pipeline.yml. The saved model can be downloaded and committed to the intended repository; '
      'this workflow does not automatically push a commit.')
    workflow = (ROOT/'.github/workflows/pipeline.yml').read_text()
    wrapped = '\n'.join('\n'.join(textwrap.wrap(line, width=105, replace_whitespace=False,
                      drop_whitespace=False) or ['']) for line in workflow.splitlines())
    story.append(Preformatted(wrapped, styles['SmallCode']))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#536675'))
        canvas.drawString(48,24,'Predictive Maintenance | Interim Report')
        canvas.drawRightString(547,24,f'Page {doc.page}')
        canvas.restoreState()
    SimpleDocTemplate(str(pdf_path), rightMargin=48, leftMargin=48, topMargin=40,
                      bottomMargin=42, title='Predictive Maintenance — Interim Report',
                      author=cfg['learner_name']).build(story, onFirstPage=footer, onLaterPages=footer)
    print('Report:',pdf_path,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--report-only',action='store_true',help='Rerender existing executed evidence without refitting models')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.get('learner_name') or cfg['learner_name']=='LearnerName':
        raise ValueError('Set learner_name before generating submission files.')
    if args.report_only:
        nb = nbformat.read(filenames(cfg)[0], as_version=4)
        if any(cell.execution_count is None for cell in nb.cells if cell.cell_type=='code'):
            raise ValueError('Notebook must be executed before report rendering.')
    else:
        nb = execute_notebook(cfg)
    render_report(cfg,nb)
    print('Notebook and HTML:',*filenames(cfg)[:2],sep='\n',flush=True)

if __name__ == '__main__':
    main()
