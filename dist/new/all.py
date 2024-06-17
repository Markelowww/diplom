import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Обновленный список log_paths
log_paths = [
    ("./new_result_log/new_result_log_xception71.tf_in1k.log", "Xception71.tf_in1k"),
    ("./new_result_log/xception41_best_0982.log", "Xception41.tf_in1k"),
    ("./new_result_log/new_result_log_wide_resnet50_2.racm_in1k.log", "Wide_resnet50_2.racm_in1k"),
    ("./new_result_log/new_result_log_vgg19_bn.tv_in1k.log", "Vgg19_bn.tv_in1k"),
    ("./new_result_log/new_result_log_resnet50.log", "ResNet-50.a1_in1k"),
    ("./new_result_log/new_result_log_xception65.tf_in1k.log", "Xception65.tf_in1k"),
    ("./new_result_log/new_result_log_tf_efficientnetv2_b3.in21k_ft_in1k.log", "Tf_efficientnetv2_b3"),
]

metrics = {
    'acc': [],
    'f1': [],
    'precision': [],
}

# Обработка лог-файлов
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

# Расчет F1-score
metrics['f1'] = [2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
                 for precision, recall in zip(metrics['precision'], metrics['acc'])]

df = pd.DataFrame(metrics)
models = [model_name for _, model_name in log_paths]

# В случае, если некоторые модели не имеют значений для всех метрик, заполнить их нулями
max_len = max(len(metrics['acc']), len(metrics['f1']), len(metrics['precision']))
for key in metrics:
    while len(metrics[key]) < max_len:
        metrics[key].append(0)

val_metrics = df.values.tolist()
sorted_indices_val = np.argsort(val_metrics, axis=0)[::-1]
models_sorted = [models[i] for i in sorted_indices_val[:, 0]]
val_metrics_sorted = np.array(val_metrics)[sorted_indices_val[:, 0]]

bar_width = 0.3
val_x = np.arange(len(models_sorted))

# Определение цветов и альфа-прозрачности
colors = ['darkblue', 'orange', 'yellow']
alphas = [1.0, 0.1, 0.1]  # Изменено значение alpha для лучшей визуализации

# Построение графика
for i, (metric_name, color, alpha) in enumerate(zip(['accuracy/recall', 'f1-score', 'precision'], colors, alphas)):
    plt.bar(val_x + i * bar_width, val_metrics_sorted[:, i], color=color, alpha=alpha, width=bar_width, label=metric_name)

plt.ylabel('Value')
plt.title('Predicted values of Xception and Inception models for test set')
plt.xticks(val_x + bar_width * (len(metrics) - 1) / 2, models_sorted, rotation=0, ha='center')  # Без наклона

plt.ylim(0.9, 1.0)

for i, metric_name in enumerate(['accuracy/recall', 'f1-score', 'precision']):
    for x, metric_value in zip(val_x, val_metrics_sorted[:, i]):
        # Округление до трех знаков без округления вверх
        metric_value = np.floor(metric_value * 1000) / 1000
        plt.text(x + i * bar_width, metric_value, f'{metric_value:.3f}', ha='center', va='bottom')

# Создание квадратика с объяснением цветов метрик
legend_elements = [plt.bar(0, 0, color=colors[i], alpha=alphas[i], label=metric_name) for i, metric_name in enumerate(['accuracy/recall', 'f1-score', 'precision'])]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(legend_elements), handlelength=4.0, handleheight=3.0)

plt.tight_layout()
plt.show()



