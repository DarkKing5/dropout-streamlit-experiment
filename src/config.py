# src/config.py

import torch

# --- Основные настройки эксперимента ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42 # Фиксируем сид для воспроизводимости

# --- Гиперпараметры модели и обучения ---
LEARNING_RATE = 1e-3 # Скорость обучения
BATCH_SIZE = 128     # Размер мини-батча
EPOCHS = 10          # Количество эпох обучения

# --- Параметры архитектуры MLP (Multi-Layer Perceptron) ---
INPUT_FEATURES = 28 * 28 # Размер изображения MNIST (784)
HIDDEN_UNITS_1 = 128     # Количество нейронов в 1-м скрытом слое
HIDDEN_UNITS_2 = 64      # Количество нейронов в 2-м скрытом слое
OUTPUT_FEATURES = 10     # Количество классов (цифры от 0 до 9)

# --- Настройки для проведения эксперимента с Dropout ---
# Мы будем запускать обучение для каждого из этих значений
DROPOUT_RATES = [0.0, 0.2, 0.5] # 0.0 - без dropout

# --- Пути для сохранения результатов ---
LOGS_DIR = "results/training_logs"
PLOTS_DIR = "results/plots"