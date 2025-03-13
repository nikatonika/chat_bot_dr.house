import torch
import numpy as np
import pickle
import faiss
import argparse
from sentence_transformers import SentenceTransformer
import os

# === Аргументы командной строки ===
parser = argparse.ArgumentParser(description="Ускоренный инференс би-энкодера Доктора Хауса")
parser.add_argument("--query", type=str, help="Текст запроса")
parser.add_argument("--hf_model", type=str, default="nikatonika/chatbot_biencoder", help="Название модели")
parser.add_argument("--triplets_path", type=str, default=os.path.join("..", "data", "house_triplets.pkl"), help="Путь к триплетам")
args = parser.parse_args()

# === Загрузка модели ===
print(f"Загрузка модели: {args.hf_model}")
model = SentenceTransformer(args.hf_model)

# === Загрузка данных ===
print(f"Загрузка триплетов из {args.triplets_path}...")
with open(args.triplets_path, "rb") as f:
    triplets_df = pickle.load(f)

house_responses = triplets_df["response"].tolist()
house_vectors = model.encode(house_responses, convert_to_numpy=True)

# === Создание индекса FAISS ===
index = faiss.IndexFlatL2(house_vectors.shape[1])
index.add(house_vectors)

# === Функция поиска ответа ===
def find_best_response(query):
    query_vector = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vector, 1)
    return house_responses[indices[0][0]]

# === Запуск ===
if args.query:
    print(f"Доктор Хаус отвечает: {find_best_response(args.query)}")
else:
    while True:
        user_query = input("Введите текст: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        print(f"Доктор Хаус отвечает: {find_best_response(user_query)}")
