import torch
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer, CrossEncoder

# Пути к данным
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESPONSES_PATH = os.path.join(DATA_DIR, 'questions_answers.npy')
VECTORS_PATH = os.path.join(DATA_DIR, 'response_embeddings.npy')

# Определение устройства
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка моделей
biencoder = SentenceTransformer("nikatonika/chatbot_biencoder_v2_cos_sim", device=device)
cross_encoder = CrossEncoder("nikatonika/chatbot_reranker_v2", device=device)

# Загрузка данных
house_responses = np.load(RESPONSES_PATH, allow_pickle=True)
response_vectors = np.load(VECTORS_PATH)
response_vectors = torch.tensor(response_vectors, dtype=torch.float32).to(device)

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'^[\'"\[]+|[\'"\]]+$', '', text)
    return text

def find_candidates(query, top_k=10):
    query_embedding = torch.tensor(
        biencoder.encode([query], convert_to_numpy=True, normalize_embeddings=True, truncate=True),
        dtype=torch.float32
    ).to(device)
    similarities = torch.matmul(response_vectors, query_embedding.T).squeeze()
    top_indices = torch.topk(similarities, k=top_k).indices
    return [clean_text(house_responses[idx.cpu().item()]) for idx in top_indices]

def rerank(query, candidates):
    if not candidates:
        return "I don't know what to say."
    pairs = [[query, c] for c in candidates]
    with torch.no_grad():
        scores = cross_encoder.predict(pairs, convert_to_numpy=True)
    best_idx = np.argmax(scores)
    best_response = candidates[best_idx] if best_idx < len(candidates) else "I don't know what to say."
    return clean_text(best_response)

def get_response(query):
    candidates = find_candidates(query, top_k=10)
    return rerank(query, candidates)

if __name__ == "__main__":
    print("\n🤖 HouseBot is ready! Type your message (or 'exit' to quit).\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = get_response(user_input)
        print(f"HouseBot: {response}\n")
