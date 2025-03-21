import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Пути к данным
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESPONSES_PATH = os.path.join(DATA_DIR, 'questions_answers.npy')
VECTORS_PATH = os.path.join(DATA_DIR, 'response_embeddings.npy')

# Определение устройства
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка модели биэнкодера
biencoder = SentenceTransformer("nikatonika/chatbot_biencoder_v2_cos_sim", device=device)

# Загрузка данных
house_responses = np.load(RESPONSES_PATH, allow_pickle=True)
response_vectors = np.load(VECTORS_PATH)
response_vectors = torch.tensor(response_vectors, dtype=torch.float32).to(device)

def find_candidates(query, top_k=10):
    query_embedding = torch.tensor(
        biencoder.encode([query], convert_to_numpy=True, normalize_embeddings=True, truncate=True),
        dtype=torch.float32
    ).to(device)

    similarities = torch.matmul(response_vectors, query_embedding.T).squeeze()
    top_indices = torch.topk(similarities, k=top_k).indices
    return [house_responses[idx.cpu().item()] for idx in top_indices]

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    candidates = find_candidates(user_input, top_k=10)
    print("\nTop candidates:")
    for i, c in enumerate(candidates, 1):
        print(f"{i}. {c}")
