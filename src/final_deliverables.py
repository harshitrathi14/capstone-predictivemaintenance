"""Execute the final notebook and build a concise, evidence-backed business report."""
import argparse
import hashlib
import json
import re
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

import nbformat
import pandas as pd
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, Preformatted

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/final'
DEST = ROOT / 'submission/final'
PREFIX = 'Harshit_Rathi_PredictiveMaintenance'
NOTEBOOK = ROOT / 'notebooks' / (PREFIX + '_Final.ipynb')
REPO = 'https://github.com/harshitrathi14/capstone-predictivemaintenance'


def read(path):
    return json.loads((ROOT / path).read_text())


def evidence():
    ev = read('evidence/final_evidence.json')
    if ev['workflow']['conclusion'] != 'success' or not ev['deployment']['verified']:
        raise ValueError('Successful hosted workflow and verified public Streamlit deployment are required.')
    if str(ev['workflow']['id']) != str(read('models/workflow_run.json')['GITHUB_RUN_ID']):
        raise ValueError('Workflow evidence does not describe the committed model.')
    if ev['deployment']['model_sha256'] != hashlib.sha256((ROOT/'models/best_model.joblib').read_bytes()).hexdigest():
        raise ValueError('Deployed model evidence is stale.')
    for name in ['github_repository.png', 'github_workflow.png', 'streamlit_cloud.png']:
        if not (ROOT/'evidence'/name).exists():
            raise FileNotFoundError(name)
    return ev


def execute_notebook(ev):
    cfg = read('config.json')
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    pipeline = (ROOT/'src/pipeline.py').read_text().split('\ndef main():')[0].replace(
        'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path.cwd()')
    analysis = (ROOT/'src/analysis.py').read_text()
    analysis = analysis.replace('Figure 5. Held-out confusion matrix', 'Figure 6. Held-out confusion matrix').replace(
        'Figure 6. Held-out ranking performance', 'Figure 5. Held-out ranking performance')
    inference = (ROOT/'src/inference.py').read_text().replace(
        'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path.cwd()')
    refs = '\n'.join(f"- [{r['title']}]({r['url']}): {r['finding']}" for r in cfg['label_evidence'])
    cells = [
        md('# Predictive Maintenance — Final Executed Notebook\n\nHarshit Rathi\n\n'
           'Companion to **Harshit_Rathi_PredictiveMaintenance_FinalReport.pdf**. '
           'This notebook contains the full analytical code, all tuning results, the committed-model evaluation, '
           'the Streamlit source, workflow source and captured deployment evidence.\n\n'
           'Run from the repository root after installing requirements.txt. All EDA uses training observations. '
           'The held-out data are never used to select model parameters or a threshold.\n\n'
           '### Dataset source and interpretation\n\n' + cfg['source'] + '\n\n' + cfg['label_note'] + '\n\n' + refs),
        md('## 1. Data registration\n\n### Methodology\nValidate the repository CSV, expected columns, numeric values, '
           'binary target, missing entries and duplicates. Record the source checksum. The functions below are the '
           'same implementation used by GitHub Actions.'),
        code(pipeline + '\nprint("Pipeline functions loaded. Seed: 42; validation and artifact checks enabled.")'),
        code('cfg = ' + repr(cfg) + '\nregistered_data, registration = validate(cfg)\nprint(json.dumps(registration, indent=2))'),
        md('## 2. Exploratory data analysis\n\n### Methodology\nPrepare a stratified 80/20 split before exploration. '
           'Describe training distributions, class comparisons and multivariate correlations. Retain outliers because '
           'physical limits and sensor units are unverified. The preparation code appears here to prevent test-data leakage; '
           'its rationale is documented separately in section 3.'),
        code("splits = Path('artifacts/final/splits')\ntrain, test = prepare(cfg, splits)\nprint('Training:', train.shape, 'Test:', test.shape)\n" +
             analysis + "\nnotes = explore(train, cfg, Path('reports/final'))"),
        md('## 3. Data preparation\n\n### Methodology\nRetain all six sensors; no unnecessary columns, missing values '
           'or duplicate rows were found. Median imputation is fitted inside each training fold. Save CSV splits and checksum '
           'metadata. Hosted preparation uploads these files, and the separate training job downloads and validates them. '
           'No timestamps or engine identifiers are available for a time-based or group-based split.'),
        code("print((splits / 'split_metadata.json').read_text())\n"
             "print('Saved split files:', sorted(p.name for p in splits.iterdir()))\n"
             "print('No rows removed; all six sensor features retained.')"),
        md('## 4. Model building with experimentation tracking\n\n### Methodology\nCompare 16 random-forest '
           'parameter combinations using three-fold stratified cross-validation. Select mean average precision (class 1), '
           'refit the best pipeline, and evaluate at the fixed 0.50 threshold. A prior classifier is the baseline. '
           'This cell performs a fresh local reproduction; the following section separately verifies the committed hosted model.'),
        code("local_metrics = train_model(cfg, splits, Path('artifacts/final/model'))\n"
             "print('Full local tuning log: all parameters, fold scores, training scores, ranks and timings')\n"
             "with pd.option_context('display.max_columns', None, 'display.max_rows', None):\n"
             "    display(pd.read_csv('artifacts/final/model/experiments.csv'))\n"
             "print(Path('artifacts/final/model/run_metadata.json').read_text())"),
        md('### Evaluation of the committed deployment model\n\nThe final report uses the model saved by GitHub Actions '
           'in `models/`, which is also loaded by Streamlit. Recompute its metrics on the same held-out split and check them '
           'against the hosted record. Show the complete hosted tuning log as well. Small platform-dependent differences '
           'between local and hosted random forests are documented rather than mixing the two sets of results.'),
        code("hosted = json.loads(Path('models/metrics.json').read_text())\n"
             "hosted_run = json.loads(Path('models/run_metadata.json').read_text())\n"
             "for name, digest in hosted_run['splits']['split_sha256'].items():\n"
             "    assert hashlib.sha256((splits / f'{name}.csv').read_bytes()).hexdigest() == digest\n"
             "committed_model = joblib.load('models/best_model.joblib')\n"
             "verified = evaluate(committed_model, test.drop(columns=cfg['target']), test[cfg['target']])\n"
             "for key, value in verified.items():\n"
             "    np.testing.assert_allclose(value, hosted['random_forest'][key], rtol=0, atol=1e-12)\n"
             "print('Committed-model metrics match the hosted record; split checksums match.')\n"
             "with pd.option_context('display.max_columns', None, 'display.max_rows', None):\n"
             "    display(pd.read_csv('models/experiments.csv'))\n"
             "diagnostics(test, cfg, Path('models'), Path('reports/final'))\n"
             "comparison = pd.DataFrame({'Local reproduction': local_metrics['random_forest'], 'Committed hosted model': hosted['random_forest']}).drop(index='confusion_matrix')\n"
             "print('Local versus hosted model metrics (full precision):')\ndisplay(comparison)\n"
             "print('Full hosted parameters:')\nprint(Path('models/best_parameters.json').read_text())\n"
             "print('Hosted lineage and environment:')\nprint(Path('models/run_metadata.json').read_text())"),
        md('## 5. Model deployment\n\n### Methodology\nLoad the committed model and its feature profile. '
           'Collect six numeric inputs into a dataframe in the training column order, validate finite numbers, and return '
           'the original dataset class and class-1 score. Defaults are rounded training medians. The app warns about inputs '
           'outside the observed training range. It does not relabel the target or claim a calibrated failure probability.'),
        code(inference + "\nmodel, profile, metadata = load_bundle()\n"
             "values = {name: round(profile['summary'][name]['median'], 2) for name in profile['features']}\n"
             "frame, predicted, score, outside = predict_observation(values, model, profile)\n"
             "display(frame)\nprint({'predicted_class': predicted, 'class_1_score': score, 'outside_training_range': outside})\n"
             "print('Model SHA-256:', profile['model_sha256'])\n"
             "Path('reports/final/sample_prediction.json').write_text(json.dumps({'inputs': values, 'class': predicted, 'score': score}, indent=2))"),
        code("from IPython.display import Markdown\n"
             "print('Complete Streamlit app source (the deployed entry point):')\n"
             "display(Markdown('```python\\n' + Path('streamlit_app.py').read_text() + '\\n```'))\n"
             "print('Deployment dependencies:')\nprint(Path('requirements.txt').read_text())\n"
             "print('Streamlit theme configuration:')\nprint(Path('.streamlit/config.toml').read_text())"),
        md('## 6. Automated GitHub Actions workflow\n\n### Methodology\nThe prepare job validates the dataset '
           'and uploads splits. The train job downloads splits, tunes/evaluates the model, tests the app with the new model '
           'and uploads the model bundle. The persist-model job commits the bundle to main. Relevant source changes '
           'and manual dispatch trigger the workflow; model-only commits do not retrigger training.'),
        code("print('Complete workflow YAML:')\nprint(Path('.github/workflows/pipeline.yml').read_text())\n"
             "print('Recorded hosted execution:')\nprint(Path('models/workflow_run.json').read_text())"),
        md('## 7. Output evaluation\n\n### Methodology\nCheck the actual repository and successful hosted workflow, '
           'then submit the same default sensor values to the public Streamlit app and compare its displayed prediction '
           'with the committed model. Screenshots are browser captures of the real pages, not reconstructed interfaces. '
           'The capture record below contains URLs, timestamps and the tested model checksum.'),
        code("from IPython.display import Image as NotebookImage\n"
             "captured = json.loads(Path('evidence/final_evidence.json').read_text())\n"
             "print(json.dumps(captured, indent=2))\n"
             "for name, label in [('github_repository.png', 'Repository folder structure'), ('github_workflow.png', 'Executed GitHub Actions workflow'), ('streamlit_cloud.png', 'Public Streamlit app prediction')]:\n"
             "    print(label)\n    display(NotebookImage(filename=str(Path('evidence') / name)))\n"
             "print('Verification checks:')\nprint(Path('evidence/test_results.txt').read_text())"),
        md('## 8. Actionable insights and recommendations\n\n'
           'The forest improves class separation over the prior baseline but has moderate discrimination and substantial errors. '
           'Its class-1 F1 at the default threshold is lower than the majority baseline; the baseline finds no class-0 observations. '
           'A business decision must therefore consider both error types.\n\n'
           '1. Confirm target meanings, pressure/temperature units and measurement context with the dataset owner.\n'
           '2. Validate unusual sensor readings rather than discarding them automatically.\n'
           '3. Obtain engine identifiers, timestamps and later outcomes; evaluate on new engines and later periods.\n'
           '4. After label confirmation, select an inspection threshold on validation data using agreed error costs.\n'
           '5. Run an analyst-reviewed pilot, monitor data quality and performance, and retain versioned model artifacts.\n\n'
           'The deployment demonstrates repeatable inference and automation. It does not establish production safety, '
           'advance warning time, remaining useful life or financial savings.\n\n'
           '### Implementation references\n'
           '- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)\n'
           '- [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html)\n'
           '- [GitHub workflow artifacts](https://docs.github.com/en/actions/tutorials/store-and-share-data)')
    ]
    nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    ExecutePreprocessor(timeout=1800, kernel_name='python3').preprocess(nb, {'metadata': {'path': str(ROOT)}})
    for c in nb.cells:
        if c.cell_type == 'code':
            if c.execution_count is None or not c.outputs or any(o.output_type == 'error' for o in c.outputs):
                raise ValueError('Every final notebook code cell must be executed with visible output and no errors.')
    nb.metadata['final_evidence'] = {'model_sha256': ev['deployment']['model_sha256'],
                                   'workflow_id': ev['workflow']['id']}
    nbformat.write(nb, NOTEBOOK)
    html, _ = HTMLExporter().from_notebook_node(nb)
    (OUT / (PREFIX + '_Notebook.html')).write_text(html)
    return nb


def workflow_diagram():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 3.1))
    ax.set(xlim=(0, 10), ylim=(0, 3)); ax.axis('off')
    labels = [('1. PREPARE', 'Validate CSV\nSave stratified splits\nUpload split artifact'),
              ('2. TRAIN', 'Download splits\nTune and evaluate\nTest app + upload model'),
              ('3. PERSIST', 'Download model bundle\nCommit to main\nRetain experiment log'),
              ('4. SERVE', 'Community Cloud\nLoad committed model\nClassify sensor inputs')]
    for i, (title, body) in enumerate(labels):
        x = .08 + 2.52*i
        ax.add_patch(FancyBboxPatch((x,.65),2.25,1.6,boxstyle='round,pad=0.06',facecolor='#edf4f5',edgecolor='#176d70'))
        ax.text(x+1.12,1.98,title,ha='center',va='center',fontsize=10,fontweight='bold',color='#17354d')
        ax.text(x+1.12,1.33,body,ha='center',va='center',fontsize=9,linespacing=1.7)
        if i < 3:
            ax.annotate('',xy=(x+2.48,1.45),xytext=(x+2.27,1.45),arrowprops={'arrowstyle':'->','color':'#176d70','lw':1.5})
    ax.text(5,.15,'Relevant push to main or manual run → artifact handoffs → model commit → app refresh',ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/'workflow_diagram.png',dpi=200,bbox_inches='tight'); plt.close(fig)


def render_report(ev):
    cfg = read('config.json')
    m = read('models/metrics.json'); rf = m['random_forest']; baseline = m['baseline']
    reg = read('artifacts/final/splits/registration.json')
    split = read('artifacts/final/splits/split_metadata.json')
    run = read('models/run_metadata.json')
    sample = read('reports/final/sample_prediction.json')
    notes = read('reports/final/eda_insights.json')
    stats = pd.read_csv(OUT/'feature_summary.csv',index_col=0)
    means = pd.read_csv(OUT/'class_comparison.csv',index_col=0)
    log = pd.read_csv(ROOT/'models/experiments.csv')
    workflow_diagram()
    styles = getSampleStyleSheet()
    styles['BodyText'].fontName='Helvetica'; styles['BodyText'].fontSize=10; styles['BodyText'].leading=14
    styles['BodyText'].spaceAfter=8
    for name in ['Heading1','Heading2','Title']:
        styles[name].textColor=colors.HexColor('#17354d')
    styles['Heading1'].fontSize=19; styles['Heading1'].leading=23; styles['Heading1'].spaceAfter=14
    styles['Heading2'].fontSize=12; styles['Heading2'].leading=16
    styles.add(ParagraphStyle('Caption',parent=styles['BodyText'],fontSize=9,leading=12,textColor=colors.HexColor('#36546b')))
    styles.add(ParagraphStyle('CodeSmall',fontName='Courier',fontSize=7,leading=9,spaceAfter=8))
    styles.add(ParagraphStyle('Cell',parent=styles['BodyText'],fontSize=8.5,leading=11,spaceAfter=0))
    story=[]
    def p(t, style='BodyText'):
        story.append(Paragraph(escape(str(t)),styles[style]))
    def h(t): p(t,'Heading1')
    def method(t): p('Methodology','Heading2'); p(t)
    def page(): story.append(PageBreak())
    def link(label,url):
        story.append(Paragraph(f'<link href="{escape(url)}" color="#176d70">{escape(label)}</link>',styles['BodyText']))
    def table(title,headers,rows,widths):
        p(title,'Heading2')
        cells=[[Paragraph(escape(str(v)),styles['Cell']) for v in row] for row in [headers]+rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6f0f1')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#176d70')),
            ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d6dfe4')),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([t,Spacer(1,9)])
    def fig(path,caption,height=310):
        im=Image(str(path)); scale=min(475/im.imageWidth,height/im.imageHeight)
        im.drawWidth=im.imageWidth*scale; im.drawHeight=im.imageHeight*scale
        story.extend([im,Spacer(1,5)]); p(caption,'Caption')
    def code(text):
        wrapped='\n'.join('\n'.join(textwrap.wrap(line, width=94, replace_whitespace=False, drop_whitespace=False)) if line else '' for line in text.splitlines())
        story.append(Preformatted(wrapped,styles['CodeSmall']))

    p('PREDICTIVE MAINTENANCE','Title')
    p('Final Project Report','Title')
    p('Harshit Rathi | AIML Capstone Project 3','Heading2')
    p('10 November 2026 | Model deployment and automation','Caption')
    story.append(Spacer(1,16))
    h('Executive overview')
    p('This project tests whether six engine sensor readings can distinguish the two recorded condition classes, '
      'then makes the selected model available through a Streamlit app and a repeatable GitHub Actions workflow. '
      'The dataset contains 19,535 observations. A stratified holdout of 3,907 observations is reserved for final evaluation.')
    p(f"The committed random forest achieves {rf['average_precision']:.2f} average precision, {rf['roc_auc']:.2f} ROC AUC "
      f"and {rf['balanced_accuracy']:.2f} balanced accuracy. The prior baseline scores {baseline['average_precision']:.2f}, "
      f"{baseline['roc_auc']:.2f} and {baseline['balanced_accuracy']:.2f}, respectively. The model finds useful associations, "
      'but substantial classification errors remain.')
    p('The public app and successful automation demonstrate that the model can be reproduced and served. '
      'The healthy/faulty label mapping and several measurement units remain unverified. Without engine identifiers, '
      'timestamps or future failure outcomes, this is condition classification rather than demonstrated advance failure prediction.')
    p('Submission files','Heading2')
    p(PREFIX+'_FinalReport.pdf','Caption'); p(PREFIX+'_Notebook.html','Caption')
    p('The PDF presents findings and business implications. The executed HTML notebook contains the full code, '
      'experiment logs and verification outputs. Raw implementation excerpts appear only in the appendices.')
    link('GitHub repository',REPO); link('Live Streamlit Community Cloud app',ev['deployment']['url'])
    page()

    h('1. Data registration and background')
    method('Load the CSV directly from the repository data folder, check all expected columns and numeric values, '
           'verify a binary target, count missing values and duplicates, and record a SHA-256 checksum. '
           'The same validation is used locally and in GitHub Actions.')
    p('Source: the supplied engine_data.csv matches Parv Modi’s Automotive Vehicles Engine Health Dataset, '
      'Kaggle version 1; the earlier source verification was recorded on 2 November 2026 [1]. The dataset documentation '
      'does not establish the measurement location, collection protocol or whether rows represent repeated engines.')
    counts=reg['class_counts']
    table('Table 1. Dataset registration summary',['Check','Finding'],[
        ['Observations / columns',f"{reg['rows']:,} / {len(reg['columns'])}"],
        ['Predictors','Engine rpm, Lub oil pressure, Fuel pressure, Coolant pressure, lub oil temp, Coolant temp'],
        ['Target','Engine Condition: original labels 0 and 1'],
        ['Class 0',f"{counts['0']:,} ({100*counts['0']/reg['rows']:.2f}%)"],
        ['Class 1',f"{counts['1']:,} ({100*counts['1']/reg['rows']:.2f}%)"],
        ['Missing values / duplicate rows',f"{sum(reg['missing'].values())} / {reg['duplicate_rows']}"],
        ['Stored source','data/engine_data.csv; checksum recorded in Appendix A']], [155,320])
    p('The target meaning requires clarification. Nafar describes class 1 as healthy [2], while Chen describes '
      'class 1 as requiring maintenance [3]. These sources conflict; the original source metadata does not resolve the mapping. '
      'Class 1 is therefore positive only for statistical evaluation, not a confirmed failure label.')
    p('Business implication: obtain the data dictionary and collection history before using a prediction to schedule '
      'maintenance. Neither the file name nor differences in sensor means establish what either class means.')
    page()

    h('2. Exploratory data analysis')
    method('Use only the 15,628 training observations. Summarize sensor distributions, compare classes and examine '
           'Pearson correlations. Flag observations beyond 1.5 interquartile ranges for review without deleting them.')
    table('Table 2. Training sensor overview',['Sensor','Mean','Std. dev.','Median','Minimum','Maximum'],
          [[name]+[f'{row[k]:.2f}' for k in ['mean','std','50%','min','max']] for name,row in stats.iterrows()], [125,70,70,70,70,70])
    p('Only rpm has an explicit unit. Pressure and temperature values retain the dataset’s original scale. '
      'Small positive pressures may round to 0.00; they are not missing values.','Caption')
    fig(OUT/'figure_1.png','Figure 1. Training class distribution; class 1 is the majority.',205)
    p(notes[0])
    p('Business implication: evaluating only overall accuracy could hide failure to recognize the smaller class. '
      'Use both-class diagnostics and compare with a prior-probability baseline.')
    page()

    h('2.1 Univariate distributions')
    fig(OUT/'figure_2.png','Figure 2. Training distributions retain their full observed ranges.',290)
    p('Engine speed and pressure measurements have upper tails. Oil temperature is concentrated near the mid-to-high '
      '70s, with a smaller high-temperature tail. A few high coolant readings stretch the displayed range, so the '
      'median and spread help describe typical observations.')
    p(notes[4])
    table('Table 3. Outlier review using the 1.5-IQR rule',['Sensor','Flagged training observations','Treatment'],
          [[n,int(r['IQR flags']),'Retained pending domain validation'] for n,r in stats.iterrows()],[155,125,195])
    p('Business implication: unusual readings could be valid operating states or measurement problems. Review '
      'calibration and units before excluding them; automatic removal could hide relevant conditions.')
    page()

    h('2.2 Bivariate class comparisons')
    fig(OUT/'figure_3.png','Figure 3. Sensor distributions overlap between the recorded classes.',285)
    table('Table 4. Mean sensor readings by class',['Sensor','Class 0','Class 1','Difference: 1 − 0'],
          [[n]+[f'{v:.2f}' for v in r] for n,r in means.iterrows()],[170,90,90,125])
    p(notes[1])
    p('Class 1 also has higher mean fuel pressure and lower mean oil temperature in this sample. '
      'The overlap supports testing combinations of sensors rather than adopting a single cutoff. '
      'These relationships describe association, not causal effects or a healthy/faulty interpretation.')
    page()

    h('2.3 Multivariate analysis')
    fig(OUT/'figure_4.png','Figure 4. Pearson correlations across the training sensors and encoded target.',355)
    p(notes[2]); p(notes[3])
    p('Business implication: no strong linear redundancy justifies dropping a sensor on correlation grounds alone. '
      'A random forest can test nonlinear combinations while retaining all available measurement context. '
      'The absence of strong linear correlations does not establish that sensors are independent.')
    page()

    h('3. Data preparation')
    method('Retain the six sensors, reject conflicting labels for identical predictor rows, and remove exact duplicates '
           'if present. Save a stratified 80/20 split with seed 42. Learn median imputation within each training fold; '
           'the test data are excluded from fitting and parameter selection.')
    table('Table 5. Preparation decisions',['Step','Decision and reason'],[
        ['Source loading','Read data/engine_data.csv directly from the repository.'],
        ['Column removal','None: no identifier or explicitly outcome-derived field is present. All six sensors are retained.'],
        ['Cleaning','No missing values or duplicate rows observed; no rows removed. Retain extreme values pending domain checks.'],
        ['Feature engineering','Use the original six readings. No unsupported unit-dependent ratios or time features are invented.'],
        ['Scaling','Not required for the tree-based model.'],
        ['Train / test split',f"{split['train_rows']:,} / {split['test_rows']:,} rows, stratified by class; seed 42."],
        ['Artifact handoff','Preparation saves train/test CSVs and metadata, then uploads the split artifact. Training downloads it and validates checksums.'],
        ['Leakage controls','Target excluded from predictors; preprocessing fitted within folds; held-out data excluded from tuning.']],[130,345])
    p('The saved split checksums agree between the notebook and the hosted workflow. This connects the report’s '
      'evaluation to the data used to produce the committed model; detailed values appear in Appendix A and the HTML notebook.')
    p('Limitation: random splitting assumes sufficiently independent observations. The file lacks engine identifiers '
      'and timestamps, so repeated-engine leakage and performance on later operating periods cannot be measured. '
      'A future operational evaluation should separate engines and time periods.')
    page()

    h('4. Model building and experiment tracking')
    method('Fit a random forest and compare it with a prior-probability dummy classifier. Search 16 parameter '
           'combinations using three shuffled stratified folds, producing 48 cross-validation fits. Select the highest '
           'mean average precision for class 1, refit the selected pipeline, and save it with all experiment records.')
    table('Table 6. Hyperparameter search',['Parameter','Candidate values'],[
        ['Number of trees','100, 200'],['Maximum depth','Unrestricted, 10'],['Minimum samples per leaf','1, 3'],
        ['Class weights','None, balanced'],['Fixed settings','Seed 42; three stratified folds; average precision for model selection']],[195,280])
    best=m['best_parameters']; bestrow=log.loc[log['rank_test_score'].idxmin()]
    p(f"Selected model: {best['model__n_estimators']} trees, maximum depth {best['model__max_depth']}, "
      f"minimum {best['model__min_samples_leaf']} sample(s) per leaf and {best['model__class_weight']} class weighting. "
      f"Mean validation average precision is {m['best_cv_average_precision']:.2f}, with fold standard deviation "
      f"{bestrow['std_test_score']:.2f}; mean training average precision is {bestrow['mean_train_score']:.2f}.")
    p('The higher training score indicates remaining overfitting. Close candidate scores do not establish that '
      'the selected model will always be best on a different sample. Depth restriction helps control complexity '
      'without assuming a linear sensor relationship.')
    p('Average precision measures how well class-1 observations are ranked across thresholds. The baseline is '
      'the class-1 prevalence. Selection by this metric does not guarantee the best F1 score or the lowest '
      'maintenance cost at the default threshold.')
    p('All candidate parameters, fold scores, ranks and timings are retained. The saved model contains both '
      'preprocessing and the fitted forest. The report’s performance figures refer to the committed hosted model '
      'used by Streamlit; the notebook also includes a separate fresh local reproduction.')
    p('Full tuning results are in Appendix B and the executed HTML notebook. Raw code and execution logs are '
      'kept outside the report’s main narrative.')
    page()

    h('4.1 Held-out model performance')
    table('Table 7. Performance on 3,907 held-out observations',['Metric','Prior baseline','Committed forest'],
          [[k.replace('_',' ').title(),f'{baseline[k]:.2f}',f'{rf[k]:.2f}'] for k in
           ['average_precision','roc_auc','balanced_accuracy','precision','recall','f1']],[235,120,120])
    fig(OUT/'figure_6.png','Figure 5. ROC and precision-recall curves for the committed model; positive class = 1.',215)
    p(f"Average precision improves by {rf['average_precision']-baseline['average_precision']:.2f} over the baseline. "
      'Balanced accuracy averages recall for both classes, so the improvement above 0.50 reflects recognition of '
      'both labels rather than majority-class guessing. ROC AUC measures ranking discrimination.')
    p(f"Class-1 F1 at threshold 0.50 is {rf['f1']:.2f}, below the baseline’s {baseline['f1']:.2f}. The baseline predicts "
      'class 1 for every observation, giving perfect class-1 recall but no class-0 recognition. The forest trades '
      'some class-1 recall for class-0 recognition. This tradeoff must be assessed against confirmed business costs.')
    page()

    h('4.2 Errors and their interpretation')
    fig(OUT/'figure_5.png','Figure 6. Confusion matrix on the held-out set. Rows are actual labels; columns are predictions.',285)
    tn,fp=rf['confusion_matrix'][0]; fn,tp=rf['confusion_matrix'][1]
    table('Table 8. Classification outcomes',['Actual class','Correct predictions','Incorrect predictions','Recall'],[
        ['Class 0',f'{tn:,}',f'{fp:,}',f'{tn/(tn+fp):.2f}'],
        ['Class 1',f'{tp:,}',f'{fn:,}',f'{tp/(tp+fn):.2f}']],[100,125,125,125])
    p(f"The model predicts class 1 for {tp+fp:,} of {split['test_rows']:,} observations "
      f"({100*(tp+fp)/split['test_rows']:.2f}%). These are dataset-class predictions, not confirmed maintenance alerts.")
    p('Neither error type can be called a missed fault or false alarm until label semantics are verified. '
      'If class 1 is later confirmed as the intervention class, its missed observations would require particular '
      'attention; if class 0 is the intervention class, the interpretation reverses. No label mapping is assumed here.')
    p('The decision threshold remains at the model default. A future threshold study should use validation data '
      'and an agreed cost for each error type, while preserving a separate test set for the final assessment.')
    page()

    h('5. Model deployment')
    method('Deploy the Streamlit entry point from the repository’s main branch on Community Cloud using Python 3.12 '
           'and the dependencies file [4]. Load the committed model once, collect six readings into a dataframe '
           'in the training feature order, and show the predicted class and class-1 model score.')
    table('Table 9. Deployment implementation',['Requirement','Implementation'],[
        ['Repository model','The app loads models/best_model.joblib, including its fitted preprocessing.'],
        ['Input capture','Six numeric inputs are stored in a one-row dataframe with the original sensor column names.'],
        ['Input checks','Reject non-finite values; warn when values are outside observed training ranges.'],
        ['Prediction','Display class 0/1 and the uncalibrated class-1 score; allow a CSV download of the assessment.'],
        ['Dependencies','requirements.txt pins the core training/inference packages to matching versions.'],
        ['Hosting','Streamlit Community Cloud; main branch; streamlit_app.py; Python 3.12.'],
        ['Model integrity','Validate the model checksum and feature schema before inference; use the repository model only.']],[140,335])
    p('Matching training and serving dependencies reduces model-loading incompatibility [5]. The model is '
      'loaded from versioned files and is not retrained for each visitor. Inputs remain in the current session '
      'unless the user downloads the assessment; this demonstration does not create a prediction log.')
    p(f"Using the displayed training-median defaults, the app predicts class {sample['class']} with a class-1 "
      f"score of {sample['score']:.2%}. This is a reproducible functional check, not a measured probability of failure.")
    link('Open the deployed app',ev['deployment']['url'])
    p('Business implication: the interface makes the model available for analyst review without requiring a '
      'notebook. Operational decisions still require verified class definitions and validation on the intended fleet.')
    page()

    h('6. Automated GitHub Actions workflow')
    method('Use separate preparation, training and model-persistence jobs in pipeline.yml. Transfer data and '
           'models as workflow artifacts [6], require upstream jobs to succeed, and commit the verified model '
           'bundle back to main using the workflow’s write permission.')
    fig(OUT/'workflow_diagram.png','Figure 7. Automated training and deployment flow. Community Cloud follows the main branch after initial setup.',165)
    table('Table 10. Workflow stages and controls',['Stage','Action / output'],[
        ['Trigger','Relevant source/data/dependency changes pushed to main, or manual workflow dispatch.'],
        ['Prepare','Install dependencies, test pipeline behavior, validate data, save and upload train/test splits.'],
        ['Train','Download splits, check hashes, tune model, evaluate holdout, save parameters, logs and provenance.'],
        ['App verification','Run Streamlit and inference tests against the newly trained model before persistence.'],
        ['Persist model','Download model artifact; commit changed model files and experiment records to main.'],
        ['Serve','After initial deployment, Community Cloud follows the selected branch and loads the committed model.']],[105,370])
    p('Model-only commits do not match the workflow’s source path filters, avoiding a training loop. '
      'Only the persistence job has repository write permission. Artifacts are retained for 90 days; '
      'the committed model, metrics and provenance remain in the repository.')
    p('Business implication: checks and explicit artifact handoffs reduce manual handling errors and connect '
      'each model to its input data and recorded experiment results. Automation does not replace model-quality review.')
    page()

    h('7. Output evaluation: repository evidence')
    method('Inspect the published repository and capture its folder structure. Verify that the data, model, '
           'workflow, dependency file and app entry point are accessible from the submitted repository link.')
    link('GitHub repository: harshitrathi14/capstone-predictivemaintenance',REPO)
    fig(ROOT/'evidence/github_repository.png','Figure 8. Actual GitHub repository folder structure captured for the final report.',345)
    table('Table 11. Repository evidence map',['Location','Purpose'],[
        ['data/','Registered engine dataset'],['src/','Validation, preparation, training and inference code'],
        ['models/','Committed model, input profile, metrics, tuning log and workflow provenance'],
        ['.github/workflows/pipeline.yml','End-to-end automation and automatic model commit'],
        ['streamlit_app.py / requirements.txt','Deployment entry point and dependencies']],[185,290])
    page()

    h('7.1 Output evaluation: executed workflow')
    link(f"Successful workflow run {ev['workflow']['id']}",ev['workflow']['url'])
    fig(ROOT/'evidence/github_workflow.png','Figure 9. Actual executed GitHub Actions workflow with successful jobs.',330)
    table('Table 12. Hosted workflow verification',['Check','Recorded result'],[
        ['Overall result',ev['workflow']['conclusion']],
        *[[j['name'],j['conclusion']] for j in ev['workflow']['jobs']],
        ['Model persistence',f"Automatic model commit {ev['workflow']['model_commit'][:12]} on main"],
        ['Artifact handoff','Train/test splits passed to training; model and experiment bundle passed to persistence']],[170,305])
    p('The notebook independently checks that the held-out split hashes and recomputed committed-model metrics '
      'agree with the hosted records. This confirms that the evaluated model and the app’s model belong to the '
      'same recorded training workflow. The local reproduction is shown separately in the HTML notebook.')
    page()

    h('7.2 Output evaluation: public Streamlit app')
    link(ev['deployment']['url'],ev['deployment']['url'])
    fig(ROOT/'evidence/streamlit_cloud.png','Figure 10. Actual public Streamlit app after submitting the default sensor readings.',360)
    p(f"The browser check submitted all six displayed defaults and returned class {sample['class']} with a "
      f"class-1 score of {sample['score']:.2%}, matching the committed model’s local inference. The public app "
      'was accessible when evidence was captured.')
    p('Seven automated checks cover invalid schema/labels, altered split artifacts, disjoint splits and model '
      'serialization, inference column order, incomplete/non-finite readings, and a Streamlit submission with '
      'an out-of-range warning. The fresh hosted model also passed the app checks before it was committed.')
    p('This evidence establishes functional deployment and reproducible inference. It does not establish '
      'production reliability, prospective failure detection or a measured reduction in maintenance cost.')
    p('Evidence captured: '+ev['captured_at']+'.','Caption')
    page()

    h('8. Actionable insights and recommendations')
    table('Table 13. Recommendations tied to findings',['Finding','Recommended action','Owner / decision'],[
        ['Label meanings and several units are unverified.','Obtain the authoritative data dictionary and confirm how labels were assigned.','Data owner + maintenance lead; prerequisite to operational interpretation.'],
        ['Sensor ranges include unusual readings.','Review calibration and original measurements; do not remove extremes solely on a statistical rule.','Sensor/domain specialist; confirm data-quality rules.'],
        ['The forest improves ranking but still makes substantial errors.','Choose an inspection threshold on validation data with agreed missed-condition and unnecessary-inspection costs.','Maintenance lead + data scientist; proceed only after label confirmation.'],
        ['No engine IDs, timestamps or future outcomes are available.','Collect these fields and test on unseen engines and later periods.','Data engineering + operations; validate generalization and warning time.'],
        ['The app and workflow make inference repeatable.','Use an analyst-reviewed pilot; monitor input ranges, class mix and later verified outcomes. Retain model versions for rollback.','ML owner; expand only when pilot results justify it.']],[125,220,130])
    p('Business value','Heading2')
    p('The immediate value is a reproducible analytical process and an accessible demonstration. These reduce '
      'the effort needed to review sensor observations and repeat model experiments. No financial saving is '
      'quantified because failure costs, intervention effectiveness and confirmed outcomes are unavailable.')
    p('Decision boundary','Heading2')
    p('Keep the current system as a supervised demonstration. Automatic maintenance orders, remaining-useful-life '
      'estimates and production claims would exceed the evidence supplied by this dataset. A prospective pilot '
      'with verified labels is the next step before operational adoption.')
    page()

    h('References')
    for i,item in enumerate(cfg['label_evidence'],1):
        p(f"[{i}] {item['title']}.")
        link(item['url'],item['url'])
        p(item['finding'],'Caption')
    for i,(title,url) in enumerate([
        ('Streamlit documentation: Deploy your app on Community Cloud','https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy'),
        ('scikit-learn documentation: Model persistence','https://scikit-learn.org/stable/model_persistence.html'),
        ('GitHub documentation: Store and share data with workflow artifacts','https://docs.github.com/en/actions/tutorials/store-and-share-data')],4):
        p(f'[{i}] {title}. Accessed 10 November 2026.'); link(url,url)
    p('The dataset is externally sourced and attributed above. Analysis, code, figures and interpretation were '
      'prepared for this project. Evidence screenshots show the project’s actual GitHub repository, hosted workflow '
      'and public app; no external stock images are used.')
    page()

    h('Appendix A. Reproducibility and evidence')
    p('The main report uses the committed hosted model. The final notebook re-executes the complete analytical '
      'pipeline locally, then independently evaluates that hosted model and verifies its recorded metrics. '
      'Different platforms may produce slightly different floating-point tree splits; local results are not substituted '
      'for the hosted model’s results.')
    table('Table A1. Hosted training environment',['Component','Version / value'],
          [['Python',run['python']]]+[[k,v] for k,v in run['packages'].items()]+
          [['Seed / folds / candidates','42 / 3 / 16'],['Inference framework','Streamlit 1.67.0'],
           ['Source commit',ev['workflow']['head_sha'][:12]],['Model commit',ev['workflow']['model_commit'][:12]]],[200,275])
    for name,digest in [('Source CSV',reg['sha256']),*split['split_sha256'].items(),('Committed model',ev['deployment']['model_sha256'])]:
        p(name+' SHA-256','Heading2'); p(digest,'Caption')
    p('Verification record','Heading2')
    p('Captured URLs, timestamps, workflow jobs and test results are displayed in notebook section 7. '
      'Full registration, split metadata, environments and model parameters are printed in notebook sections 1–5. '
      'Full code and outputs are included there to keep this report readable.')
    page()

    h('Appendix B. Extended experiment results')
    p('Table B1 reports all 16 hosted candidates. Extra decimal precision is retained here to distinguish close '
      'scores. Full fold scores, timings and training scores are displayed in notebook section 4 and saved in '
      'models/experiments.csv. The notebook also includes the separate fresh local tuning run.')
    table('Table B1. Complete hosted cross-validation comparison',['Trees','Depth','Leaf','Weight','CV AP','CV std.','Rank'],
          [[int(r['param_model__n_estimators']), 'None' if pd.isna(r['param_model__max_depth']) else int(r['param_model__max_depth']),
            int(r['param_model__min_samples_leaf']), 'None' if pd.isna(r['param_model__class_weight']) else r['param_model__class_weight'],
            f"{r['mean_test_score']:.5f}",f"{r['std_test_score']:.5f}",int(r['rank_test_score'])] for _,r in log.iterrows()],
          [48,48,42,85,94,94,64])
    p('AP = average precision for class 1; CV = cross-validation. Rank is based on mean validation AP. '
      'The held-out test set is not used to rank candidates. The saved best_parameters.json also records '
      'all fixed estimator and preprocessing parameters.')
    page()

    h('Appendix C. Implementation excerpts')
    p('These excerpts document the deployed inference and artifact handoff. The complete analytical implementation '
      'is executed in the HTML notebook; complete app source and workflow YAML are displayed in sections 5 and 6.')
    p('C1. Validate inputs and classify an observation','Heading2')
    code('def predict_observation' + (ROOT/'src/inference.py').read_text().split('def predict_observation',1)[1])
    p('C2. Pass prepared data between workflow jobs','Heading2')
    code('''# Preparation job
- uses: actions/upload-artifact@v4
  with:
    name: train-test-splits
    path: artifacts/splits/
# Training job
- uses: actions/download-artifact@v4
  with:
    name: train-test-splits
    path: artifacts/splits/
- name: Tune and evaluate using downloaded split artifacts
  run: python src/pipeline.py train''')
    p('C3. Model persistence','Heading2')
    p('After app verification, the persist-model job downloads the model bundle, stages models/, commits changed '
      'files with the workflow run ID and pushes to main. Its exact executable shell steps and successful execution '
      'record appear in the HTML notebook. No raw logs are embedded in the business narrative.')

    def footer(canvas,doc):
        canvas.saveState(); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#536778'))
        canvas.drawString(45,25,'Harshit Rathi | Predictive Maintenance | Final Report')
        canvas.drawRightString(550,25,f'Page {doc.page}'); canvas.restoreState()
    path=OUT/(PREFIX+'_FinalReport.pdf')
    doc=SimpleDocTemplate(str(path),pagesize=(595.28,841.89),rightMargin=55,leftMargin=55,
                          topMargin=43,bottomMargin=44,title='Predictive Maintenance — Final Report',author='Harshit Rathi')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); DEST.mkdir(parents=True,exist_ok=True)
    ev=evidence()
    if not args.report_only:
        execute_notebook(ev)
    else:
        nb=nbformat.read(NOTEBOOK,4)
        if nb.metadata['final_evidence']['model_sha256'] != ev['deployment']['model_sha256']:
            raise ValueError('The committed model changed; execute the notebook again.')
    pdf=render_report(ev)
    for path in [pdf,OUT/(PREFIX+'_Notebook.html')]:
        shutil.copy2(path,DEST/path.name)
    manifest={'prepared_at':datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
              'workflow_id':ev['workflow']['id'],'app_url':ev['deployment']['url'],
              'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in DEST.iterdir() if p.suffix in ['.pdf','.html']}}
    (OUT/'submission_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
