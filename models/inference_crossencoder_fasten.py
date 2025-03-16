import torch
import numpy as np
import argparse
from sentence_transformers import CrossEncoder
import os

# === Аргументы командной строки ===
parser = argparse.ArgumentParser(description="Ускоренный инференс кросс-энкодера Доктора Хауса")
parser.add_argument("--query", type=str, help="Текст запроса")
parser.add_argument("--hf_model", type=str, default="nikatonika/chatbot_reranker_v2", help="Название модели")
parser.add_argument("--candidates_path", type=str, default=os.path.join("data", "questions_answers.npy"), help="Путь к кандидатам")
parser.add_argument("--similarities_path", type=str, default=os.path.join("data", "cosine_similarities.npy"), help="Путь к предрасчитанным косинусным близостям")
args = parser.parse_args()

# === Загрузка модели ===
print(f"Загрузка модели: {args.hf_model}")
cross_encoder = CrossEncoder(args.hf_model, device="cuda" if torch.cuda.is_available() else "cpu")

# === Загрузка предрасчитанных данных ===
print(f"Загрузка ответов из {args.candidates_path}...")
house_responses = np.load(args.candidates_path, allow_pickle=True)

print(f"Загрузка матрицы косинусных близостей из {args.similarities_path}...")
cosine_similarities = np.load(args.similarities_path)

# === Функция ранжирования ===
def rerank_with_cross_encoder(query):
    """Выбирает лучший ответ с помощью предрасчитанных косинусных близостей и кросс-энкодера"""
    query_scores = cross_encoder.predict([[query, candidate] for candidate in house_responses])
    best_idx = np.argmax(query_scores)
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
