import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, ROUND_DOWN

# Updated list of log_paths
log_paths = [
    ("./prezenta/optimizer/xception71_base.log", "base"),
    ("./prezenta/optimizer/xception71_SGD_0.8.log", "xception71_SGD_0.8"),
    ("./prezenta/optimizer/xception71_SGD_0.9_false.log", "xception71_SGD_0.9_false"),
    ("./prezenta/optimizer/xception71_SGD_0.9_true.log", "xception71_SGD_0.9_true"),
    ("./prezenta/optimizer/xception71_SGD_0.85.log", "xception71_SGD_0.85"),
    ("./prezenta/optimizer/xception71_SGD_0.95.log", "xception71_SGD_0.95"),
    ("./prezenta/optimizer/xception71_SGD_0.99.log", "xception71_SGD_0.99"),
    ("./prezenta/optimizer/xception71_AdamW_[0.9, 0.999].log", "xception71_AdamW_[0.9, 0.999]"),
    ("./prezenta/optimizer/xception71_AdamW_[0.85, 0.999].log", "xception71_AdamW_[0.85, 0.999]"),
    ("./prezenta/optimizer/xception71_AdamW_[0.92, 0.999].log", "xception71_AdamW_[0.92, 0.999]"),

]

metrics = {
    'acc': [],
    'f1': [],
    'precision': [],
}

# Process log files
for path, _ in log_paths:
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                metric_name = parts[0]
                metric_value = float(parts[1])
                if metric_name in metrics:
                    metrics[metric_name].append(metric_value)

# Calculate F1 score
metrics['f1'] = [
    2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    for precision, recall in zip(metrics['precision'], metrics['acc'])
]

# Ensure all metrics have the same length
max_len = max(len(metrics['acc']), len(metrics['f1']), len(metrics['precision']))
for key in metrics:
    while len(metrics[key]) < max_len:
        metrics[key].append(0)

df = pd.DataFrame(metrics)
models = [model_name for _, model_name in log_paths]
val_metrics = df.values.tolist()
sorted_indices_val = np.argsort(val_metrics, axis=0)[::-1]
models_sorted = [models[i] for i in sorted_indices_val[:, 0]]
val_metrics_sorted = np.array(val_metrics)[sorted_indices_val[:, 0]]

bar_width = 0.3
val_x = np.arange(len(models_sorted))

# Define colors and alphas
colors = ['darkblue', 'orange', 'yellow']
alphas = [1.0, 0.1, 0.1]

# Helper function to format values without rounding up
def format_decimal(value, decimal_places):
    decimal_value = Decimal(value).quantize(Decimal('1.' + '0' * decimal_places), rounding=ROUND_DOWN)
    return str(decimal_value)

# Plotting
for i, (metric_name, color, alpha) in enumerate(zip(['accuracy/recall', 'f1-score', 'precision'], colors, alphas)):
    plt.bar(val_x + i * bar_width, val_metrics_sorted[:, i], color=color, alpha=alpha, width=bar_width, label=metric_name)

plt.ylabel('Value')
plt.title('Predicted values of Xception and Inception models for test set')
plt.xticks(val_x + bar_width * (len(metrics) - 1) / 2, models_sorted, rotation=45, ha='right')
plt.ylim(0.9, 1.0)
for i, metric_name in enumerate(['accuracy/recall', 'f1-score', 'precision']):
    for x, metric_value in zip(val_x, val_metrics_sorted[:, i]):
        formatted_value = format_decimal(metric_value, 3)
        plt.text(x + i * bar_width, metric_value, formatted_value, ha='center', va='bottom')

# Create legend elements
legend_elements = [plt.bar(0, 0, color=colors[i], alpha=alphas[i], label=metric_name) for i, metric_name in enumerate(['accuracy/recall', 'f1-score', 'precision'])]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(legend_elements), handlelength=2.0, handleheight=2.0)

plt.tight_layout()
plt.show()
