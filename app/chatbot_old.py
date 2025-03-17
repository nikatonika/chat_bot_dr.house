import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from scipy.spatial.distance import cdist

# Определение устройства
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Загрузка моделей ===
print("Загрузка биэнкодера...")
try:
    bi_encoder = SentenceTransformer("nikatonika/chatbot_biencoder", device=device)
except Exception as e:
    print(f"Ошибка загрузки биэнкодера: {e}")
    exit(1)

print("Загрузка кросс-энкодера...")
try:
    cross_encoder = CrossEncoder("nikatonika/chatbot_reranker", device=device)
except Exception as e:
    print(f"Ошибка загрузки кросс-энкодера: {e}")
    exit(1)

# === Поиск данных ===
data_paths = ["data", "data_old"]  # Ищем сначала в data, потом в data_old
response_vectors_path = None
house_responses_path = None

for path in data_paths:
    if os.path.exists(os.path.join(path, "response_vectors.pkl")):
        response_vectors_path = os.path.join(path, "response_vectors.pkl")
    if os.path.exists(os.path.join(path, "house_responses.npy")):
        house_responses_path = os.path.join(path, "house_responses.npy")

if not response_vectors_path or not house_responses_path:
    print("Ошибка: файлы эмбеддингов не найдены ни в data, ни в data_old!")
    exit(1)

print(f"Загрузка эмбеддингов из {response_vectors_path} и {house_responses_path}...")
try:
    response_vectors = np.load(response_vectors_path, allow_pickle=True)
    house_responses = np.load(house_responses_path, allow_pickle=True)
except Exception as e:
    print(f"Ошибка загрузки эмбеддингов: {e}")
    exit(1)

# === Функции инференса ===
def rerank_with_cross_encoder(query, candidates):
    """Ранжирует кандидатов с помощью кросс-энкодера"""
    if isinstance(candidates, np.ndarray):
        candidates = candidates.tolist()
    pairs = [[query, str(candidate)] for candidate in candidates]  
    scores = cross_encoder.predict(pairs)  
    best_idx = np.argmax(scores)
    return candidates[best_idx]

def get_house_response(query):
    """Выдает ответ, используя биэнкодер для поиска и кросс-энкодер для ранжирования"""
    query_embedding = bi_encoder.encode([query], convert_to_numpy=True)

    # Поиск ближайших векторов
    distances = cdist(query_embedding, response_vectors, metric="cosine")
    best_indices = np.argsort(distances[0])[:10]  

    candidates = [house_responses[idx] for idx in best_indices]

    best_response = rerank_with_cross_encoder(query, candidates)
    return best_response
