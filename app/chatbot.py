import numpy as np
import torch
import pickle
from sentence_transformers import SentenceTransformer, util
import os

# === Загрузка обученной модели ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("nikatonika/chatbot_biencoder", device=device)

# === Загрузка предрассчитанных векторов ответов ===
data_path = "data/response_vectors.pkl"
if os.path.exists(data_path):
    with open(data_path, "rb") as f:
        response_vectors = pickle.load(f)
else:
    raise FileNotFoundError(f"Файл {data_path} не найден. Проверь путь и наличие данных.")

# === Загрузка текстов ответов ===
triplets_path = "data/house_triplets.pkl"
if os.path.exists(triplets_path):
    with open(triplets_path, "rb") as f:
        triplets_df = pickle.load(f)
    house_responses = triplets_df["response"].tolist()
else:
    raise FileNotFoundError(f"Файл {triplets_path} не найден. Проверь путь и наличие данных.")

# === Функция поиска наиболее похожего ответа ===
def get_house_response(user_input: str):
    """Находит наиболее подходящий ответ в базе."""
    query_vector = model.encode([user_input], convert_to_numpy=True)
    scores = util.cos_sim(query_vector, response_vectors)[0]
    best_match_idx = np.argmax(scores)
    return house_responses[best_match_idx]

if __name__ == "__main__":
    while True:
        query = input("Введите текст: ")
        if query.lower() in ["exit", "quit"]:
            break
        print(f"Доктор Хаус отвечает: {get_house_response(query)}")