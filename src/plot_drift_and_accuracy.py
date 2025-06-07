import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Global plotting config
mpl.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 26,
    'legend.fontsize': 18
})

# Colors
color_drift = '#d95f02'          # Orange
color_adaptimizer = '#1b5e9e'    # Blue
color_adamw = '#a6cee3'          # Light Blue
color_static = 'goldenrod'       # Gold

# Configuration
data_type = 'finetune'  # or 'pretrain'
fs = (12, 8)
windows = 203
pretrain_start = 20
results_path = 'results/stream/'
file_name = 'metrics.csv'
mode = 3  # 1 = single run, 2 = fixed model, 3 = all models

# Model sets
models = {
    'finetuned': [
        'albert_10yrs_finetuned', 'bert_10yrs_finetuned', 'deberta_10yrs_finetuned',
        'distilbert_10yrs_finetuned', 'electra_10yrs_finetuned', 'roberta_10yrs_finetuned'
    ],
    'adaptimizer': [
        'albert_10yrs_finetuned_adaptimizer', 'bert_10yrs_finetuned_adaptimizer', 'deberta_10yrs_finetuned_adaptimizer',
        'distilbert_10yrs_finetuned_adaptimizer', 'electra_10yrs_finetuned_adaptimizer', 'roberta_10yrs_finetuned_adaptimizer'
    ],
    'static': [
        'albert_10yrs', 'deberta_10yrs', 'electra_10yrs'
    ]
} if data_type == 'finetune' else {
    'finetuned': [
        'PRETRAIN_albert_10yrs_finetuned_bs_16_ws_10', 'PRETRAIN_bert_10yrs_finetuned_bs_32_ws_10',
        'PRETRAIN_deberta_10yrs_finetuned_bs_8_ws_10', 'PRETRAIN_distilbert_10yrs_finetuned_bs_64_ws_10',
        'PRETRAIN_electra_10yrs_finetuned_bs_16_ws_10', 'PRETRAIN_roberta_10yrs_finetuned_bs_32_ws_10'
    ],
    'adaptimizer': [
        'PRETRAIN_albert_10yrs_finetuned_bs_16_ws_10_adaptimizerdm_cosine_dt_2',
        'PRETRAIN_bert_10yrs_finetuned_bs_32_ws_10_adaptimizerdm_cosine_dt_2',
        'PRETRAIN_deberta_10yrs_finetuned_bs_8_ws_10_adaptimizerdm_cosine_dt_2',
        'PRETRAIN_distilbert_10yrs_finetuned_bs_64_ws_10_adaptimizer',
        'PRETRAIN_electra_10yrs_finetuned_bs_16_ws_10_adaptimizerdm_cosine_dt_2',
        'PRETRAIN_roberta_10yrs_finetuned_bs_32_ws_10_adaptimizerdm_cosine_dt_2'
    ],
    'static': [
        'PRETRAIN_albert_10yrs', 'PRETRAIN_deberta_10yrs', 'PRETRAIN_electra_10yrs'
    ]
}

# Year ticks and article counts for nonlinear time axis
years_labels = np.array(range(2015, 2026))
article_counts = np.array([54928, 46782, 40477, 38878, 37609, 39513, 37337, 34580, 32172, 35751, 13268])
tick_positions = np.insert(np.cumsum(article_counts) / np.sum(article_counts), 0, 0)

# Helper functions
def read_metrics(models_list):
    return [pd.read_csv(f'{results_path}{model}/{file_name}') for model in models_list]

def get_windowed(df, column, windows=20):
    return [chunk.mean() for chunk in np.array_split(df[column], windows)]

def get_prequential_accuracy(df):
    df['prequential_accuracy'] = (
        (df['accuracy'] * df['step'] - df['accuracy'].shift(1) * df['step'].shift(1)) /
        (df['step'] - df['step'].shift(1))
    )
    df['prequential_accuracy'] = df['prequential_accuracy'].rolling(2, min_periods=1).mean().dropna()
    return df

# Load model metrics
dfs_adamw = read_metrics(models['finetuned'])
dfs_adaptimizer = read_metrics(models['adaptimizer'])
dfs_static = read_metrics(models['static'])

# Load drift data
drift_path = 'results/drift_scores_finetuned.csv' \
    if data_type == 'finetune' else \
    'results/drift_scores_pretrained.csv'
df_drift = pd.read_csv(drift_path)
if data_type == 'pretrain':
    df_drift = df_drift.iloc[pretrain_start:].reset_index(drop=True)
df_drift['drift_score'] = df_drift['drift_score'].astype(float)
drift_scores = get_windowed(df_drift, 'drift_score', windows=windows)

# Process prequential accuracies
accuracies_adamw_all, accuracies_adapt_all, accuracies_static_all = [], [], []

for df_a, df_b, df_s in zip(dfs_adamw, dfs_adaptimizer, dfs_static):
    for df in [df_a, df_b, df_s]:
        df['accuracy'] *= 100
        if data_type == 'pretrain':
            df.drop(df.index[:pretrain_start], inplace=True)
            df.reset_index(drop=True, inplace=True)
        get_prequential_accuracy(df)

    accuracies_adamw_all.append(get_windowed(df_a, 'prequential_accuracy', windows))
    accuracies_adapt_all.append(get_windowed(df_b, 'prequential_accuracy', windows))
    accuracies_static_all.append(get_windowed(df_s, 'prequential_accuracy', windows))

# Compute means
mean_adamw = np.mean(accuracies_adamw_all, axis=0)
mean_adapt = np.mean(accuracies_adapt_all, axis=0)
mean_static = np.mean(accuracies_static_all, axis=0)

# ============ Plot ============ #
fig, ax1 = plt.subplots(figsize=fs)
x = np.linspace(0, 1, windows)

# Plot drift
ax1.plot(x, drift_scores, label='Drift Score', color=color_drift)
ax1.axhline(y=np.mean(drift_scores), color=color_drift, linestyle='--', label='Mean Drift Score')
ax1.tick_params(axis='y', labelcolor=color_drift)
ax1.set_xlabel('Year')

# Plot accuracy
ax2 = ax1.twinx()
ax2.plot(x, mean_adapt, label='OFA-Adaptimizer', color=color_adaptimizer)
ax2.plot(x, mean_adamw, label='OFA-AdamW', color=color_adamw, linestyle='--')
if data_type == 'finetune':
    ax2.plot(x, mean_static, label='Static-Finetuned', color=color_static, linestyle='--')
else:
    ax2.plot(x, mean_static, label='Static-Pretrained', color=color_static, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color_adaptimizer)

# Final plot tweaks
ax1.legend(*ax1.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0.0, 1), fontsize=12)
plt.xticks(tick_positions, labels=years_labels, rotation=45)
plt.tight_layout()

# Save
output_base = f'figures/drift_score_plot_{data_type}'
for ext in ['png', 'eps', 'pdf']:
    plt.savefig(f'{output_base}.{ext}')
plt.show()