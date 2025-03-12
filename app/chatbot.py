import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from scipy.spatial.distance import cdist

# Определение устройства
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Загрузка моделей ===
print("Загрузка биэнкодера...")
bi_encoder = SentenceTransformer("nikatonika/chatbot_biencoder", device=device)

print("Загрузка кросс-энкодера...")
cross_encoder = CrossEncoder("nikatonika/chatbot_reranker", device=device)

# === Загрузка эмбеддингов ответов ===
print("Загрузка эмбеддингов ответов...")
response_vectors = np.load("data/response_vectors.pkl", allow_pickle=True)
house_responses = np.load("data/house_responses.npy", allow_pickle=True)

# === Функции инференса ===
def rerank_with_cross_encoder(query, candidates):
    """Ранжирует кандидатов с помощью кросс-энкодера"""
    pairs = [[query, candidate] for candidate in candidates]
    scores = cross_encoder.predict(pairs)  # теперь predict работает!
    best_idx = np.argmax(scores)
    return candidates[best_idx]

def get_house_response(query):
    """Выдает ответ, используя биэнкодер для поиска и кросс-энкодер для ранжирования"""
    query_embedding = bi_encoder.encode([query], convert_to_numpy=True)

    # Поиск ближайших векторов без faiss
    distances = cdist(query_embedding, response_vectors, metric="cosine")
    best_indices = np.argsort(distances[0])[:5]  # Берем 5 ближайших кандидатов

    candidates = [house_responses[idx] for idx in best_indices]

    # Ранжируем с помощью кросс-энкодера и выбираем лучший ответ
    best_response = rerank_with_cross_encoder(query, candidates)

    return best_response
