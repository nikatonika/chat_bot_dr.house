# ChatBot Dr. House

Чат-бот, который отвечает репликами Доктора Хауса, используя **retrieval-based** подход с биэнкодером и кросс-энкодером для ранжирования ответов. Проект включает предобработку данных, обучение моделей, инференс и веб-сервис на FastAPI.

![Чат-бот Доктора Хауса](static/Screenshot.png)

## 📌 Используемые датасеты

### Основные данные
Скрипты диалогов из различных сериалов использовались для разделения на **героя** (Доктор Хаус) и **антигероев** (другие персонажи). Данные загружены с **Kaggle** и **Hugging Face**:

- **Доктор Хаус**: [Kaggle](https://www.kaggle.com/datasets/milozampari/house-md)
- **Антагонисты**:
  - [Game of Thrones](https://www.kaggle.com/datasets/gopinath15/gameofthrones)
  - [Breaking Bad](https://www.kaggle.com/datasets/mexwell/breakingbad-script)
  - [Futurama](https://www.kaggle.com/datasets/arianmahin/the-futurama-dataset)
  - [Attack on Titan (Eren Jaeger)](https://www.kaggle.com/datasets/gauriket/erenjeager)
  - [The Office](https://www.kaggle.com/datasets/vkaul3/the-office-entire-script-for-nlp-applications)
  - [Star Wars](https://www.kaggle.com/datasets/xvivancos/star-wars-movie-scripts)
  - [Rick & Morty](https://www.kaggle.com/datasets/andradaolteanu/rickmorty-scripts)
  - [Friends](https://www.kaggle.com/datasets/blessondensil294/friends-tv-series-screenplay-script)
  - [The Big Bang Theory](https://www.kaggle.com/datasets/mitramir5/the-big-bang-theory-series-transcript)
  - [SpongeBob](https://huggingface.co/datasets/krplt/spongebob_transcripts)

---

## 🔹 Подготовка данных

### **1. Объединение датасетов**
Все данные приведены к единому формату CSV:
- **house_final_cleaned.csv** – реплики Доктора Хауса
- **antagonists_final_cleaned.csv** – реплики антагонистов

### **2. Очистка и фильтрация**
- Удалены ненужные символы и пробелы
- Фильтрация коротких реплик (<10 символов)
- Разбиение длинных реплик на чанки (до 30 слов)
- Исключение дубликатов

### **3. Генерация триплетов**
Создан датасет **house_triplets.pkl**, содержащий:
- **Анкер** (вопрос или реплика)
- **Позитивный ответ** (реплика Доктора Хауса)
- **Негативный ответ** (реплика антагонистов, семантически отличающаяся)

---

## 🏋️ Обучение моделей

### **1. Обучение биэнкодера**
Использовалась модель **distilroberta-base**, дообученная на триплетах:
- **Transformer** – энкодер
- **Pooling** – агрегация эмбеддингов
- **Triplet Loss** – функция потерь

#### **Процесс обучения**
- Разделение данных: **train (80%) / validation (20%)**
- 8 эпох, **batch_size=8**
- Использование **Triplet Loss**
- **Learning rate = 2e-5**
- Проверка на валидации

#### **Результаты**
Показатель **cosine similarity**:
- **До обучения**: ~0.21
- **После обучения**: ~0.83

### **2. Обучение кросс-энкодера**
Использовалась модель **roberta-base**, дообученная на кросс-энкодерном датасете:
- На вход передаются пары **(вопрос, ответ)**
- Модель предсказывает **релевантность ответа**
- Функция потерь: **Binary Cross-Entropy**

#### **Результаты**
- **Accuracy**: 86% на валидации
- **Лучшие ответы** стали более естественными

---

## 🚀 Инференс

### **1. Поиск кандидатов (биэнкодер)**
Используется **inference_biencoder.py**
- Загружается **предобученный биэнкодер**
- Находится **TOP-K** релевантных ответов

```bash
python models/inference_biencoder.py --query "When will i feel better?"

{"response": "Take your time. Well continue with the medicine."}

2. Ранжирование (кросс-энкодер)
Используется inference_crossencoder.py

Берет TOP-K кандидатов
Ранжирует их
Выдает лучший ответ

3. Ускоренный инференс
Используется inference_biencoder_fasten.py

Применяет FAISS для быстрого поиска
Ускоряет инференс в ~6 раз

🎛 Веб-интерфейс (FastAPI)

1. Запуск сервера
Чат-бот реализован на FastAPI, UI – на Jinja2

2. Вызов API
curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'Content-Type: application/json' \
  -d '{"text": "hello house?"}'

  📁 Структура проекта

  chat_bot_dr.house/
│── app/
│   ├── main.py               # FastAPI сервер
│   ├── chatbot.py            # Логика ответов
│   ├── utils.py              # Вспомогательные функции
│── data/
│   ├── house_final_cleaned.csv  # Данные Доктора Хауса
│   ├── antagonists_final_cleaned.csv  # Данные антагонистов
│   ├── house_triplets.pkl    # Триплеты для обучения
│   ├── response_vectors.pkl  # Векторные представления ответов
│── data_preparation/
│   ├── data_for_cross_encoder_preparing.ipynb # Подготовка данных для кросс-энкодера
│   ├── data_for_cross_encoder_preparing.ipynb # Подготовка данных для кросс-энкодера
│   ├── datasets_merging.py   # Объединение датасетов
│── models/
│   ├── bi_encoder_training.ipynb   # Обучение биэнкодера
│   ├── cross_encoder_training.ipynb   # Обучение кросс-энкодера
│   ├── inference_biencoder.py   # Инференс биэнкодера
│   ├── inference_biencoder_fasten.py  # Ускоренный инференс биэнкодера
│   ├── inference_crossencoder.py  # Инференс кросс-энкодера
│── templates/
│   ├── index.html            # Веб-интерфейс
│── static/
│   ├── style.css             # Стили UI
│── logs/
│   ├── debug.log             # Логи сервера
│── README.md  
│── requirements.txt          # Зависимости проекта


⚡ Оптимизация инференса
Использован FAISS для быстрого поиска ближайших эмбеддингов
Предварительное кэширование эмбеддингов ответов
Оптимизирована обработка запросов

🔧 Установка и запуск

# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
uvicorn app.main:app --reload

📝 Итог
Чат-бот Доктора Хауса создан с использованием retrieval-based подхода. Были проведены: ✅ Очистка данных
✅ Генерация триплетов
✅ Обучение моделей
✅ Оптимизация инференса
✅ Реализация API и веб-интерфейса

Проект полностью соответствует требованиям и готов к использованию. 
