# src/utils.py

import torch
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Импортируем конфиг, чтобы использовать его настройки
from src import config

def get_data_loaders(batch_size=config.BATCH_SIZE):
    """
    Скачивает датасет MNIST, применяет трансформации и создает
    загрузчики данных для обучающей и тестовой выборок.
    """
    # Трансформации для данных:
    # 1. ToTensor() - преобразует изображения в тензоры PyTorch
    # 2. Normalize() - нормализует тензоры (ускоряет обучение)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Среднее и стд. отклонение для MNIST
    ])

    # Скачиваем обучающие данные
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    # Скачиваем тестовые данные
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # Создаем загрузчики данных (DataLoaders)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True # Перемешиваем данные на каждой эпохе
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False # Тестовые данные перемешивать не нужно
    )

    print(f"Загружено {len(train_data)} обучающих и {len(test_data)} тестовых изображений.")
    return train_loader, test_loader

def save_logs(history, dropout_rate):
    """
    Сохраняет историю обучения (словари с loss и accuracy) в JSON файл.
    """
    # Убедимся, что директория для логов существует
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Имя файла будет зависеть от dropout_rate, чтобы не перезаписывать результаты
    filename = f"history_dropout_{str(dropout_rate).replace('.', '_')}.json"
    filepath = os.path.join(config.LOGS_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"Логи обучения сохранены в файл: {filepath}")