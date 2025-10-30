import gradio as gr
import pandas as pd
import joblib
import os

# Загружаем модель
model = joblib.load("model.pkl")

def predict_survival(pclass, sex, age, fare):
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [1 if sex == "Женщина" else 0],
        'Age': [age],
        'Fare': [fare]
    })
    pred = model.predict(data)
    return "💚 Выжил" if pred[0] == 1 else "💀 Не выжил"

# Интерфейс
interface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Number(label="Класс каюты (1–3)"),
        gr.Radio(["Мужчина", "Женщина"], label="Пол"),
        gr.Number(label="Возраст"),
        gr.Number(label="Цена билета (£)")
    ],
    outputs="text",
    title="Titanic Survival Predictor",
    description="Введи данные пассажира и узнай, выжил ли он на Титанике 🚢"
)

# Добавляем эту строку ↓↓↓
port = int(os.environ.get("PORT", 7860))
interface.launch(server_name="0.0.0.0", server_port=port)
