"""Training-only exploratory analysis and held-out diagnostic plots."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay


def explore(train, cfg, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = cfg['target']
    features = train.drop(columns=target).columns.tolist()
    sns.set_theme(style='whitegrid', font_scale=0.95)
    summary = train[features].describe().T
    q1, q3 = summary['25%'], summary['75%']
    iqr = q3 - q1
    outlier_counts = ((train[features] < q1 - 1.5*iqr) | (train[features] > q3 + 1.5*iqr)).sum()
    summary['IQR flags'] = outlier_counts
    groups = train.groupby(target)[features].mean().T
    groups.columns = ['Class 0 mean', 'Class 1 mean']
    groups['Difference (1 - 0)'] = groups['Class 1 mean'] - groups['Class 0 mean']
    corr = train.corr()
    summary.to_csv(output / 'feature_summary.csv')
    groups.to_csv(output / 'class_comparison.csv')
    corr.to_csv(output / 'correlations.csv')
    print('Table 1. Training feature overview; pressure/temperature units were not supplied.')
    display(summary.round(2))
    print('Table 2. Training feature means by class (associations, not causal effects).')
    display(groups.round(2))
    print('Table 3. Training missing values and IQR outlier flags (flags are retained).')
    display(pd.DataFrame({'Missing': train[features].isna().sum(), 'IQR flags': outlier_counts}))

    def finish(fig, number):
        fig.tight_layout()
        fig.savefig(output / f'figure_{number}.png', dpi=180, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    counts = train[target].value_counts().sort_index()
    ax.bar(['Class 0', 'Class 1'], counts, color=['#536d91', '#25a399'])
    for i, count in enumerate(counts):
        ax.text(i, count, f'{count:,} ({count/len(train):.2%})', ha='center', va='bottom')
    ax.set(ylim=(0, counts.max()*1.18), xlabel='Engine Condition', ylabel='Training observations',
           title='Figure 1. Training class distribution')
    finish(fig, 1)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for col, ax in zip(features, axes.flat):
        ax.hist(train[col], bins=35, color='#536d91', edgecolor='white')
        ax.set(xlabel=col, ylabel='Observations')
    fig.suptitle('Figure 2. Sensor distributions (training data)', fontsize=14)
    finish(fig, 2)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for col, ax in zip(features, axes.flat):
        sns.boxplot(data=train, x=target, y=col, ax=ax, color='#8ac9c2', fliersize=2)
        ax.set(xlabel='Engine Condition class', ylabel=col)
    fig.suptitle('Figure 3. Sensor distributions by class (training data)', fontsize=14)
    finish(fig, 3)

    fig, ax = plt.subplots(figsize=(9, 6.6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0, vmin=-1, vmax=1,
                ax=ax, annot_kws={'size':10})
    ax.set_title('Figure 4. Multivariate Pearson correlations (training data)')
    finish(fig, 4)

    target_corr = corr[target].drop(target)
    strongest = target_corr.abs().idxmax()
    upper_triangle = np.triu(np.ones((len(features), len(features)), dtype=bool), k=1)
    feature_corr = corr.loc[features, features].where(upper_triangle).stack()
    pair = feature_corr.abs().idxmax()
    rpm_means = groups.loc['Engine rpm']
    coolant = summary.loc['Coolant temp']
    notes = [
        f'Class 1 represents {100*train[target].mean():.2f}% of training observations. A majority-class prediction can therefore appear accurate while never recognizing class 0; balanced accuracy and both-class diagnostics are needed.',
        f'Engine rpm averages {rpm_means.iloc[0]:.2f} for class 0 and {rpm_means.iloc[1]:.2f} for class 1. The distributions overlap, so rpm alone should not be used as a maintenance rule.',
        f'{strongest} has the strongest absolute linear association with the target (r = {target_corr[strongest]:.2f}). This describes an association with the encoded class, not a cause of engine condition.',
        f'The strongest sensor-to-sensor correlation is {pair[0]} versus {pair[1]} (r = {feature_corr[pair]:.2f}). Weak pairwise correlations do not rule out nonlinear relationships or interactions.',
        f'Coolant temp has a median of {coolant["50%"]:.2f} and a maximum of {coolant["max"]:.2f}; {int(outlier_counts["Coolant temp"])} readings lie outside its 1.5-IQR limits. Retain these readings because they may describe unusual operating conditions; verify units and sensor calibration before applying physical limits.',
        'No identifier, timestamp, or failure-mode field is present. All six sensor features are retained. These measurements support contemporaneous condition classification; the data cannot establish advance warning time or remaining useful life.'
    ]
    (output / 'eda_insights.json').write_text(json.dumps(notes, indent=2))
    for note in notes:
        print(note)
    return notes


def diagnostics(test, cfg, model_dir, output):
    output = Path(output)
    model = joblib.load(Path(model_dir) / 'best_model.joblib')
    metrics = json.loads((Path(model_dir) / 'metrics.json').read_text())
    y = test[cfg['target']]
    X = test.drop(columns=cfg['target'])
    probability = model.predict_proba(X)[:, 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(metrics['random_forest']['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0','Class 1'], yticklabels=['Class 0','Class 1'], ax=ax)
    ax.set(xlabel='Predicted class', ylabel='Actual class', title='Figure 5. Held-out confusion matrix')
    fig.tight_layout(); fig.savefig(output / 'figure_5.png', dpi=180); plt.show(); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    RocCurveDisplay.from_predictions(y, probability, ax=axes[0], name='Random forest')
    axes[0].plot([0,1],[0,1], '--', color='grey', label='Chance'); axes[0].legend()
    PrecisionRecallDisplay.from_predictions(y, probability, ax=axes[1], name='Random forest')
    axes[1].axhline(y.mean(), linestyle='--', color='grey', label='Class 1 prevalence'); axes[1].legend()
    fig.suptitle('Figure 6. Held-out ranking performance (positive class = 1)')
    fig.tight_layout(); fig.savefig(output / 'figure_6.png', dpi=180); plt.show(); plt.close(fig)
    print('Table 5. Held-out metrics (positive class = 1).')
    display(pd.DataFrame({name:{k:v for k,v in metrics[name].items() if k!='confusion_matrix'}
                          for name in ['baseline','random_forest']}).round(2))
    print(f"Held-out balanced accuracy: {metrics['random_forest']['balanced_accuracy']:.2f}; "
          f"baseline: {metrics['baseline']['balanced_accuracy']:.2f}. "
          f"Class-1 F1: {metrics['random_forest']['f1']:.2f}; baseline: {metrics['baseline']['f1']:.2f}.")
    print('Selection optimized cross-validated average precision, not F1 at threshold 0.5. '
          'The majority baseline has class-1 recall of 1.00 but recognizes no class-0 observations. '
          'The forest trades some class-1 recall for class-0 recognition; threshold choice needs validation data and operational costs.')
