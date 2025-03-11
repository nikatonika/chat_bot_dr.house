import torch
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
import argparse

# === Аргументы командной строки ===
parser = argparse.ArgumentParser(description="Инференс модели Доктора Хауса")
parser.add_argument("--query", type=str, help="Текст запроса (анкор)")
parser.add_argument("--hf_model", type=str, default="nikatonika/chatbot_biencoder", help="Название модели на Hugging Face")
parser.add_argument("--triplets_path", type=str, default="data/house_triplets.pkl", help="Путь к триплетам")
args = parser.parse_args()

# === Загрузка модели ===
print(f"Загружаем модель с Hugging Face: {args.hf_model}")
model = SentenceTransformer(args.hf_model)

# === Загрузка триплетов (база ответов) ===
print(f"Загружаем триплеты из {args.triplets_path}...")
with open(args.triplets_path, "rb") as f:
    triplets_df = pickle.load(f)

# Используем только позитивные ответы
house_responses = triplets_df["response"].tolist()

# === Векторизация всех ответов (кэшируем для быстрого инференса) ===
print("Векторизуем базу ответов...")
house_vectors = model.encode(house_responses, convert_to_numpy=True)

# === Функция поиска лучшего ответа ===
def find_best_response(query):
    """Находит наиболее похожий ответ в базе."""
    query_vector = model.encode([query], convert_to_numpy=True)
    scores = util.cos_sim(query_vector, house_vectors)[0]
    best_idx = np.argmax(scores)
    return house_responses[best_idx]

# === Запуск инференса ===
if args.query:
    response = find_best_response(args.query)
    print(f"Доктор Хаус отвечает: {response}")
else:
    while True:
        query = input("Введите текст: ")
        if query.lower() in ["exit", "quit"]:
            break
        print(f"Доктор Хаус отвечает: {find_best_response(query)}")
