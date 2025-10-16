# src/model.py

import torch.nn as nn

# Импортируем конфиг, чтобы использовать его настройки
from src import config

class MLP(nn.Module):
    """
    Класс для многослойного перцептрона (MLP) с двумя скрытыми слоями
    и возможностью применения Dropout.
    """
    def __init__(self, dropout_rate=0.0):
        super(MLP, self).__init__()

        # Мы используем nn.Sequential для удобного и чистого определения
        # последовательности слоёв.
        self.layers = nn.Sequential(
            # 1-й слой: преобразует входные 784 фичи в 128
            nn.Linear(config.INPUT_FEATURES, config.HIDDEN_UNITS_1),
            nn.ReLU(),
            # Слой Dropout применяется после активации.
            # Если dropout_rate = 0.0, он ничего не делает.
            nn.Dropout(p=dropout_rate),

            # 2-й слой: преобразует 128 фич в 64
            nn.Linear(config.HIDDEN_UNITS_1, config.HIDDEN_UNITS_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # Выходной слой: преобразует 64 фичи в 10 (для 10 классов)
            nn.Linear(config.HIDDEN_UNITS_2, config.OUTPUT_FEATURES)
        )

    def forward(self, x):
        """
        Определяет прямой проход данных через сеть.
        """
        # Входной тензор 'x' имеет размер (batch_size, 1, 28, 28).
        # Мы его "выпрямляем" в (batch_size, 784), чтобы подать в nn.Linear.
        x = x.view(x.size(0), -1)
        
        # Прогоняем выпрямленный тензор через наши слои
        return self.layers(x)