import gradio as gr
import pandas as pd
import joblib

# Загружаем модель
model = joblib.load("model.pkl")

# Функция для предсказания
def predict_survival(pclass, sex, age, fare):
    # Преобразуем ввод пользователя в формат, который был при обучении
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [1 if sex == "Женщина" else 0],
        'Age': [age],
        'Fare': [fare]
    })
    
    # Делаем предсказание
    prediction = model.predict(data)
    return "💚 Выжил" if prediction[0] == 1 else "💀 Не выжил"

# Создаем интерфейс
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

# Запуск приложения
interface.launch()
