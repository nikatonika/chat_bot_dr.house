import torch
import numpy as np
import faiss
import argparse
from sentence_transformers import SentenceTransformer
import os

# === Аргументы командной строки ===
parser = argparse.ArgumentParser(description="Ускоренный инференс би-энкодера Доктора Хауса с FAISS")
parser.add_argument("--query", type=str, help="Текст запроса")
parser.add_argument("--hf_model", type=str, default="nikatonika/chatbot_biencoder_v2_cos_sim", help="Название модели")
parser.add_argument("--embeddings_path", type=str, default=os.path.join("data", "response_embeddings.npy"), help="Путь к эмбеддингам")
parser.add_argument("--responses_path", type=str, default=os.path.join("data", "questions_answers.npy"), help="Путь к ответам")
args = parser.parse_args()

# === Загрузка модели ===
print(f"Загрузка модели: {args.hf_model}")
model = SentenceTransformer(args.hf_model, device="cuda" if torch.cuda.is_available() else "cpu")

# === Загрузка данных ===
print(f"Загрузка эмбеддингов из {args.embeddings_path}...")
house_vectors = np.load(args.embeddings_path)

print(f"Загрузка ответов из {args.responses_path}...")
house_responses = np.load(args.responses_path, allow_pickle=True)

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
