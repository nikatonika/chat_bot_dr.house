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
    bi_encoder = SentenceTransformer("nikatonika/chatbot_biencoder_v2_cos_sim", device=device)
except Exception as e:
    print(f"Ошибка загрузки биэнкодера: {e}")
    exit(1)

print("Загрузка кросс-энкодера...")
try:
    cross_encoder = CrossEncoder("nikatonika/chatbot_reranker_v2", device=device)
except Exception as e:
    print(f"Ошибка загрузки кросс-энкодера: {e}")
    exit(1)

# === Поиск данных ===
data_folder = "data"
response_vectors_path = os.path.join(data_folder, "response_embeddings.npy")
house_responses_path = os.path.join(data_folder, "questions_answers.npy")
sarcasm_responses_path = os.path.join(data_folder, "house_sarcasm.npy")
sarcasm_embeddings_path = os.path.join(data_folder, "house_sarcasm_embeddings.npy")

# Проверяем существование файлов
if not all(os.path.exists(path) for path in [
    response_vectors_path, house_responses_path,
    sarcasm_responses_path, sarcasm_embeddings_path
]):
    print("Ошибка: файлы эмбеддингов не найдены!")
    exit(1)

print(f"Загрузка эмбеддингов из {data_folder}...")
try:
    response_vectors = np.load(response_vectors_path, allow_pickle=True)
    house_responses = np.load(house_responses_path, allow_pickle=True)
    sarcasm_responses = np.load(sarcasm_responses_path, allow_pickle=True)
    sarcasm_embeddings = np.load(sarcasm_embeddings_path, allow_pickle=True)
except Exception as e:
    print(f"Ошибка загрузки эмбеддингов: {e}")
    exit(1)

# === Настройки поиска ===
TOP_K = 10

def find_candidates_fast(queries, top_k=10):
    """Оптимизированный поиск кандидатов с batch-инференсом."""
    query_embeddings = bi_encoder.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    similarities = np.dot(response_vectors, query_embeddings.T)
    
    top_indices = np.argpartition(-similarities, top_k, axis=0)[:top_k].T  

    batch_candidates = [
    [" ".join(str(house_responses[idx]).split("\n")).strip("[]'\"") for idx in indices]
    for indices in top_indices
    ]

    # Логирование
    print(f"Кандидаты перед ранжированием (исправленные): {batch_candidates}")

    return batch_candidates

def rerank_with_cross_encoder_fast(query, candidates):
    """Оптимизированный кросс-энкодер с batch-инференсом."""
    if not candidates or all(len(c) == 0 for c in candidates):
        return get_sarcastic_response(query)  # Если нет кандидатов, берем заглушку

    # Разворачиваем вложенные массивы и удаляем `repr()`
        # Если строка содержит несколько реплик в одном элементе, берем только первую
    candidates = [c.split('"  "')[0].strip() for c in candidates]
    candidates = [str(c).strip("[]'\"") for c in candidates]
    batch_pairs = [[query, candidate] for candidate in candidates]

    print(f"Запрос: {query}")
    print(f"Кандидаты после исправления: {batch_pairs}")

    with torch.no_grad():
        scores = cross_encoder.predict(batch_pairs, convert_to_numpy=True)

    best_idx = np.argmax(scores)
    best_response = candidates[best_idx].strip("[]'\"") if best_idx < len(candidates) else get_sarcastic_response(query)

    # Проверяем, что мы взяли только одну строку, а не несколько
    if "\n" in best_response:
        best_response = best_response.split("\n")[0]  # Берем только первую строку

    best_response = best_response.strip()



    print(f"Финальный ответ: {best_response}")

    return best_response

def get_sarcastic_response(query):
    """Выбирает саркастическую заглушку."""
    query_embedding = bi_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sarcasm_distances = cdist(query_embedding, sarcasm_embeddings, metric="cosine")[0]

    best_sarcasm_idx = np.argmin(sarcasm_distances)
    return str(sarcasm_responses[best_sarcasm_idx])


def get_house_response_fast(queries):
    """Получает финальный ответ на основе оптимизированного пайплайна с обработкой пустых кандидатов."""
    batch_candidates = find_candidates_fast(queries, top_k=10)

    if all(len(c) == 0 for c in batch_candidates):
        return [get_sarcastic_response(q) for q in queries]

    responses = [rerank_with_cross_encoder_fast(q, c).strip("[]'\"") for q, c in zip(queries, batch_candidates)]

    return [resp if isinstance(resp, str) else str(resp) for resp in responses]  # Гарантируем строки







