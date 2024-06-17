import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log_paths = [
    ("result_log/test_results_resnet50.log", "ResNet-50"),
    ("result_log/test_results_resnet18.fb_swsl_ig1b_ft_in1k.log", "Resnet18.fb_swsl_ig1b_ft_in1k"),
    ("result_log/test_results_resnet152.log", "ResNet-152"),
    ("result_log/test_results_resnet34.log", "ResNet-34"),
    ("result_log/test_results_vit_base_r50_s16_224.orig_in21k.log", "ViT-Base R50 S16 224"),
    ("result_log/test_results_repvgg_a1.rvgg_in1k.log", "RepVGG-A1"),
]

metrics = {
    'acc': [],
    'f1': [],
    'precision': [],
    'recall': []
}

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

df = pd.DataFrame(metrics)
models = [model_name for _, model_name in log_paths]
val_metrics = df.values.tolist()
sorted_indices_val = np.argsort(val_metrics, axis=0)[::-1]
models_sorted = [models[i] for i in sorted_indices_val[:, 0]]
val_metrics_sorted = np.array(val_metrics)[sorted_indices_val[:, 0]]

bar_width = 0.2
val_x = np.arange(len(models))

colors = ['orange', 'green', 'purple', 'blue']
for i, metric_name in enumerate(metrics.keys()):
    plt.bar(val_x + i * bar_width, val_metrics_sorted[:, i], color=colors[i], width=bar_width, label=metric_name)

plt.ylabel('Value')
plt.title('Best predicted values of all models for test set')
plt.xticks(val_x + bar_width * (len(metrics) - 1) / 2, models_sorted, rotation=45, ha='right')

for i, metric_name in enumerate(metrics.keys()):
    for x, metric_value in zip(val_x, val_metrics_sorted[:, i]):
        plt.text(x + i * bar_width, metric_value, f'{metric_value:.3f}', ha='center', va='bottom')

plt.legend()
plt.tight_layout()
plt.show()
