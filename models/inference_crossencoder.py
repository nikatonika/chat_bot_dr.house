import torch
import numpy as np
import pickle
import argparse
from sentence_transformers import CrossEncoder
import os

# === Аргументы командной строки ===
parser = argparse.ArgumentParser(description="Инференс кросс-энкодера Доктора Хауса")
parser.add_argument("--query", type=str, help="Текст запроса")
parser.add_argument("--hf_model", type=str, default="nikatonika/chatbot_reranker", help="Название модели")
parser.add_argument("--candidates_path", type=str, default=os.path.join("..", "data", "house_responses.npy"), help="Путь к кандидатам")
args = parser.parse_args()

# === Загрузка модели ===
print(f"Загрузка модели: {args.hf_model}")
cross_encoder = CrossEncoder(args.hf_model)

# === Загрузка кандидатов ===
print(f"Загрузка ответов из {args.candidates_path}...")
house_responses = np.load(args.candidates_path, allow_pickle=True)

# === Функция ранжирования ===
def rerank_with_cross_encoder(query):
    pairs = [[query, candidate] for candidate in house_responses]
    scores = cross_encoder.predict(pairs)
    best_idx = np.argmax(scores)
    return house_responses[best_idx]

# === Запуск ===
if args.query:
    print(f"Доктор Хаус отвечает: {rerank_with_cross_encoder(args.query)}")
else:
    while True:
        user_query = input("Введите текст: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        print(f"Доктор Хаус отвечает: {rerank_with_cross_encoder(user_query)}")
