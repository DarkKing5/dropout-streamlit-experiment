# app.py

import streamlit as st
import torch
from src.model import MLP
from src.utils import get_data_loaders # Можем использовать для примеров
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas
import json
import os

# --- Настройки страницы и темы ---
st.set_page_config(
    page_title="Dropout Analysis | by Dinozawr & Madya",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Кастомный CSS для футуристической серой темы ---
st.markdown("""
<style>
    .reportview-container {
        background: #2E2E2E;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background: #1C1C1C;
    }
    h1, h2, h3 {
        color: #00A8E8; /* Яркий акцентный цвет */
    }
    .stButton>button {
        border: 2px solid #00A8E8;
        border-radius:20px;
        color: #00A8E8;
        background-color: transparent;
    }
    .stButton>button:hover {
        border-color: #FFFFFF;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


# --- Заголовок ---
st.title("🔬 Анализ влияния Dropout на переобучение")
st.caption("Проект выполнили: Dinozawr и Madya")

# --- Боковая панель (Sidebar) ---
st.sidebar.header("Навигация")
app_mode = st.sidebar.selectbox(
    "Выберите раздел:",
    ["📈 Дашборд результатов", "🎨 Интерактивное распознавание"]
)

# --- ОСНОВНАЯ ЛОГИКА ---

# Функция для загрузки модели
@st.cache_resource
def load_model(path):
    model = MLP(dropout_rate=0.2) # Загружаем структуру
    model.load_state_dict(torch.load(path, map_location='cpu')) # Загружаем веса
    model.eval()
    return model

# Загружаем нашу лучшую модель
best_model = load_model("models/model_dropout_0_2.pth")


# --- Раздел 1: Дашборд ---
if app_mode == "📈 Дашборд результатов":
    st.header("Сравнение моделей с разным уровнем Dropout")
    st.image("results/plots/comparison_plot.png")

    st.subheader("Итоговые метрики")
    # Тут можно вставить таблицу из README
    
    # TODO: Добавить интерактивный выбор модели и показ её индивидуальных графиков

# --- Раздел 2: Интерактивное распознавание ---
elif app_mode == "🎨 Интерактивное распознавание":
    st.header("Нарисуйте цифру от 0 до 9")

    # Создаем холст
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заполнения
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Распознать цифру"):
        if canvas_result.image_data is not None:
            # 1. Предобработка изображения
            img = canvas_result.image_data.astype(np.uint8)
            img = Image.fromarray(img).convert('L')
            
            # Трансформации как при обучении
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            img_tensor = transform(img).unsqueeze(0)

            # 2. Предсказание
            with torch.no_grad():
                output = best_model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                prediction = torch.argmax(probabilities).item()
                confidence = probabilities[prediction].item()

            # 3. Вывод результата
            st.subheader(f"Я думаю, это цифра: **{prediction}**")
            st.write(f"Уверенность: {confidence:.2%}")

            # 4. График уверенности
            st.bar_chart(probabilities)