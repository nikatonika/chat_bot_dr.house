import torch
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import argparse

# === Аргументы командной строки ===
parser = argparse.ArgumentParser(description="Инференс модели Доктора Хауса")
parser.add_argument("--query", type=str, help="Текст запроса (анкор)")
parser.add_argument("--hf_model", type=str, default="nikatonika/chatbot_biencoder", help="Название модели на Hugging Face")
parser.add_argument("--vector_path", type=str, default="data/response_vectors.pkl", help="Путь к векторизованным ответам")
parser.add_argument("--triplets_path", type=str, default="data/house_triplets.pkl", help="Путь к триплетам")
args = parser.parse_args()

# === Загрузка модели с Hugging Face ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Загружаем модель с Hugging Face: {args.hf_model}")
model = SentenceTransformer(args.hf_model, device=device)

# === Загрузка предрассчитанных векторов ===
print("Загружаем предрассчитанные векторы ответов...")
with open(args.vector_path, "rb") as f:
    response_vectors = pickle.load(f)

# Преобразуем в float32 (так требует FAISS)
response_vectors = response_vectors.astype(np.float32)

# === Загрузка текстов ответов ===
print("Загружаем тексты ответов...")
with open(args.triplets_path, "rb") as f:
    triplets_data = pickle.load(f)

house_responses = triplets_data["response"].tolist()

# === Инициализация FAISS ===
index = faiss.IndexFlatIP(response_vectors.shape[1])
index.add(response_vectors)

# === Функция поиска лучшего ответа ===
def find_best_response(query):
    """Находит наиболее похожий ответ в базе с FAISS."""
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)
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
