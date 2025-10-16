# src/visualize.py

import json
import os
import matplotlib.pyplot as plt

from src import config

def plot_results():
    """
    Читает все файлы логов из папки results/training_logs,
    строит сравнительные графики loss и accuracy и сохраняет их.
    """
    print("\n--- Построение графиков по результатам экспериментов ---")
    
    # Убедимся, что директория для графиков существует
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # Создаем два поля для графиков: один для Loss, другой для Accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Настраиваем цвета для разных линий
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Проходим по всем заданным dropout rates
    for i, rate in enumerate(config.DROPOUT_RATES):
        # Формируем путь к файлу логов
        log_filename = f"history_dropout_{str(rate).replace('.', '_')}.json"
        log_filepath = os.path.join(config.LOGS_DIR, log_filename)

        if not os.path.exists(log_filepath):
            print(f"Файл логов не найден: {log_filepath}. Пропускаем...")
            continue

        # Читаем историю из JSON
        with open(log_filepath, 'r') as f:
            history = json.load(f)

        epochs = range(1, len(history['train_loss']) + 1)
        label_prefix = f'Dropout p={rate}'
        color = colors[i % len(colors)]

        # --- График Loss ---
        ax1.plot(epochs, history['train_loss'], linestyle='--', color=color, label=f'{label_prefix} (Train Loss)')
        ax1.plot(epochs, history['val_loss'], linestyle='-', color=color, label=f'{label_prefix} (Val Loss)')
        
        # --- График Accuracy ---
        ax2.plot(epochs, history['train_acc'], linestyle='--', color=color, label=f'{label_prefix} (Train Acc)')
        ax2.plot(epochs, history['val_acc'], linestyle='-', color=color, label=f'{label_prefix} (Val Acc)')

    # --- Оформление графика Loss ---
    ax1.set_title('Сравнение Loss в зависимости от Dropout')
    ax1.set_xlabel('Эпохи')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- Оформление графика Accuracy ---
    ax2.set_title('Сравнение Accuracy в зависимости от Dropout')
    ax2.set_xlabel('Эпохи')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Сохраняем итоговый график в файл
    fig.tight_layout() # Чтобы графики не наезжали друг на друга
    plot_path = os.path.join(config.PLOTS_DIR, "comparison_plot.png")
    plt.savefig(plot_path)
    
    print(f"Итоговый график сохранен в: {plot_path}")