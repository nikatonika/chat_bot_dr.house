# Chatbot Tyrion - Обучение модели

## 📌 Описание
Этот проект представляет собой чат-бота, обученного на репликах Тириона Ланнистера из *Game of Thrones*. Модель основана на **Sentence-BERT** (`all-MiniLM-L6-v2`) и дообучается на диалогах персонажа.

## 📂 Структура проекта
```
chat_bot_tirion/
│── app/                     # Исходный код бота
│── data/                    # Исходные данные и эмбеддинги
│   ├── game-of-thrones.csv   # Исходные реплики Тириона
│   ├── tyrion_embeddings.pkl # Сохраненные эмбеддинги
│── models/                   # Сохраненные веса модели
│   ├── fine_tuned_tyrion_model/
│── static/                   # Статические файлы (если нужны)
│── templates/                # HTML шаблоны (если нужны)
│── train_model.py            # Скрипт обучения модели
│── chatbot.py                # Основной скрипт работы бота
│── requirements.txt          # Зависимости проекта
│── README.md                 # Документация проекта
```

## 🚀 Установка и запуск
### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Запуск обучения модели
```bash
python train_model.py
```

### 3. Запуск чат-бота
```bash
python chatbot.py
```

## 📊 Данные
- Исходные данные: `data/game-of-thrones.csv`
- Файл с эмбеддингами: `data/tyrion_embeddings.pkl`
- Обученная модель сохраняется в `models/fine_tuned_tyrion_model`

## 🛠️ Основные библиотеки
- `sentence-transformers`
- `torch`
- `pandas`
- `matplotlib`

## 🔧 Автор
Nikatonika
