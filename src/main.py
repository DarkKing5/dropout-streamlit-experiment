# src/main.py

from src import train
from src import config
from src import visualize

def main():
    """
    Основная функция для запуска всего конвейера:
    1. Обучает модели для каждого значения dropout_rate.
    2. Строит и сохраняет итоговые графики.
    """
    print(">>> Начало полного цикла эксперимента <<<")

    # 1. Запускаем обучение для каждого dropout rate из конфига
    for rate in config.DROPOUT_RATES:
        train.train_model(dropout_rate=rate)

    # 2. После всех тренировок, строим графики
    visualize.plot_results()

    print("\n>>> Эксперимент полностью завершен! <<<")
    print(f"Результаты сохранены в папках: {config.LOGS_DIR} и {config.PLOTS_DIR}")

# Стандартная конструкция в Python для запуска основного кода
if __name__ == "__main__":
    main()