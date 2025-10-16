# src/train.py

import torch
import torch.nn as nn
import os
from tqdm import tqdm # Библиотека для красивых progress bar

# Импортируем наши модули
from src import config
from src import model as model_module
from src import utils

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    """
    Проводит одну эпоху обучения модели.
    """
    model.train() # Переводим модель в режим обучения
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Оборачиваем data_loader в tqdm для отображения прогресса
    loop = tqdm(data_loader, leave=True)
    for inputs, targets in loop:
        # Перемещаем данные на указанное устройство (CPU или GPU)
        inputs, targets = inputs.to(device), targets.to(device)

        # 1. Прямой проход (Forward pass)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # 2. Обратный проход и оптимизация (Backward pass and optimization)
        optimizer.zero_grad() # Обнуляем градиенты
        loss.backward()       # Вычисляем градиенты
        optimizer.step()      # Обновляем веса

        # Собираем статистику
        total_loss += loss.item() * inputs.size(0)
        _, predictions = torch.max(outputs, 1)
        correct_predictions += (predictions == targets).sum().item()
        total_samples += inputs.size(0)
        
        # Обновляем progress bar
        loop.set_postfix(loss=loss.item(), acc=correct_predictions / total_samples)

    avg_loss = total_loss / total_samples
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc

def validate_one_epoch(model, data_loader, loss_fn, device):
    """
    Проводит одну эпоху валидации модели.
    """
    model.eval() # Переводим модель в режим оценки
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # torch.no_grad() отключает вычисление градиентов, ускоряя процесс
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Прямой проход
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Собираем статистику
            total_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc


def train_model(dropout_rate):
    """
    Основная функция, которая запускает и координирует процесс обучения
    и валидации для модели с заданным dropout_rate.
    """
    print(f"\n--- Начало обучения для Dropout Rate: {dropout_rate} ---")

    # Устанавливаем сид для воспроизводимости
    torch.manual_seed(config.RANDOM_SEED)

    # 1. Загружаем данные
    train_loader, test_loader = utils.get_data_loaders()

    # 2. Инициализируем модель, функцию потерь и оптимизатор
    model = model_module.MLP(dropout_rate=dropout_rate).to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 3. Словарь для хранения истории обучения
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # 4. Цикл обучения по эпохам
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, config.DEVICE
        )
        val_loss, val_acc = validate_one_epoch(
            model, test_loader, loss_fn, config.DEVICE
        )

        print(
            f"Эпоха {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Записываем результаты эпохи в историю
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    # 5. Сохраняем логи
    utils.save_logs(history, dropout_rate)

    # 6. Сохраняем веса модели
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"model_dropout_{str(dropout_rate).replace('.', '_')}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Веса модели сохранены в: {model_path}")

    print(f"--- Обучение для Dropout Rate: {dropout_rate} завершено ---")
    return model, history