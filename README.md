# ChatBot Dr. House

Чат-бот, который ведёт диалог в стиле Доктора Хауса, используя retrieval-based подход и биэнкодер для поиска наиболее релевантных ответов. Проект включает предобработку данных, обучение модели и веб-сервис на FastAPI.

---

## 📌 Используемые датасеты

### Основные данные:
Были собраны скрипты диалогов из различных сериалов. Они использовались для разделения на **героя** (Доктор Хаус) и **антигероев** (другие персонажи). Данные загружены с **Kaggle** и **Hugging Face**:

- **Доктор Хаус**: [Kaggle](https://www.kaggle.com/datasets/milozampari/house-md)
- **Антигерои**:
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

## 🔹 Подготовка данных

### 🔽 **1. Объединение датасетов**
Все данные были конвертированы в единый формат CSV. Были выделены два отдельных датасета:
1. **house_final_dataset.csv** – реплики Доктора Хауса
2. **antagonists_chunks_dataset.csv** – реплики антагонистов

### 🔽 **2. Очистка и фильтрация**
- Удаление незначимых символов и лишних пробелов
- Фильтрация слишком коротких и бессмысленных реплик (\<10 символов)
- Разбиение длинных реплик на чанки (до 30 слов)
- Исключение повторяющихся строк

### 🔽 **3. Генерация триплетов**
Для обучения биэнкодера был создан датасет **house_triplets.pkl**, содержащий:
- **Анкер** (вопрос или реплика)
- **Позитивный ответ** (реплика Доктора Хауса)
- **Негативный ответ** (реплика антагонистов, наиболее отличная по смыслу)

## 🏋️ Обучение биэнкодера

### 🔽 **1. Выбор архитектуры**
Была использована модель **distilroberta-base** с дообучением. Архитектура состоит из:
- **Transformer** – энкодер входных реплик
- **Pooling** – агрегация эмбеддингов
- **TripletLoss** – функция потерь для обучения

### 🔽 **2. Процесс обучения**
- Разделение данных на **train** (80%) и **validation** (20%)
- 8 эпох обучения с **batch_size=8**
- Использование **Triplet Loss**
- Оптимизация learning rate (**2e-5**)
- Проверка на валидационном наборе

### 🔽 **3. Оценка качества**
Показатель **cosine similarity** между анкорами и позитивными ответами:
- **До обучения**: ~0.21
- **После обучения**: ~0.83

## 🚀 Инференс (FastAPI)

### 🔽 **1. Запуск модели**
Чат-бот загружает предобученный **биэнкодер** и находит наиболее релевантные ответы с помощью FAISS.

**Пример вызова API:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Ты же врач?"}'
```
**Ответ:**
```json
{"response": "No, treating illnesses is why we became doctors."}
```

### 🔽 **2. Веб-интерфейс**
Веб-страница с UI написана с использованием **Jinja2**. Чтобы запустить:
```bash
uvicorn app.main:app --reload
```

## ⚡ Ускорение инференса

- Использован **FAISS** для быстрого поиска ближайших эмбеддингов
- Предварительное кэширование эмбеддингов ответов
- Оптимизированы размер контекста и длина реплик

## 📁 Структура проекта
```plaintext
chat_bot_dr.house/
│── app/
│   ├── main.py           # FastAPI сервер
│   ├── chatbot.py        # Логика обработки запросов
│   ├── utils.py          # Вспомогательные функции
│── data/
│   ├── house_final_dataset.csv
│   ├── antagonists_chunks_dataset.csv
│   ├── house_triplets.pkl
│── models/
│   ├── bi_encoder_training.ipynb
│   ├── inference_biencoder.py
│── templates/
│   ├── index.html
│── README.md
│── requirements.txt
```

## 🔧 Установка и запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
uvicorn app.main:app --reload
```

## 📝 Итог
Чат-бот Доктора Хауса был создан с использованием retrieval-based подхода и обученного биэнкодера. Были проведены несколько этапов улучшения, включая:
- Очистку данных
- Генерацию триплетов
- Обучение модели
- Оптимизацию инференса
- Реализацию API и веб-интерфейса

Проект соответствует всем требованиям задания и готов к использованию.

