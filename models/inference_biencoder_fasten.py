import torch
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer, util
import argparse

# === Аргументы командной строки ===
parser = argparse.ArgumentParser(description="Инференс модели Доктора Хауса")
parser.add_argument("--query", type=str, help="Текст запроса (анкор)")
parser.add_argument("--hf_model", type=str, default="nikatonika/chatbot_biencoder", help="Название модели на Hugging Face")
parser.add_argument("--vector_path", type=str, default="data/response_vectors.pkl", help="Путь к векторизованным ответам")
args = parser.parse_args()

# === Загрузка модели с Hugging Face ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Загружаем модель с Hugging Face: {args.hf_model}")
model = SentenceTransformer(args.hf_model, device=device)

# === Загрузка векторизованных ответов ===
print("Загружаем предрассчитанные векторы ответов...")
with open(args.vector_path, "rb") as f:
    response_vectors = pickle.load(f)

# === Загрузка текстов ответов ===
with open("data/house_responses.pkl", "rb") as f:
    house_responses = pickle.load(f)

# === Инициализация FAISS ===
index = faiss.IndexFlatIP(response_vectors.shape[1])
index.add(response_vectors)

# === Функция поиска лучшего ответа ===
def find_best_response(query):
    """Находит наиболее похожий ответ в базе с FAISS."""
    query_vector = model.encode([query], convert_to_numpy=True)
    _, best_idx = index.search(query_vector, 1)
    return house_responses[best_idx[0][0]]

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
