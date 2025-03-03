import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import pandas as pd

# Загружаем обученную модель
model = SentenceTransformer("fine_tuned_tyrion_model")

# Загружаем данные
file_path = "data/game-of-thrones.csv"
df = pd.read_csv(file_path)

# Фильтруем реплики Тириона
tyrion_lines = df[df["Speaker"] == "TYRION"]["Text"].dropna().tolist()

# Кеширование эмбеддингов
embeddings_file = "tyrion_embeddings.pkl"

if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        tyrion_embeddings = pickle.load(f)
else:
    tyrion_embeddings = model.encode(tyrion_lines, show_progress_bar=True)
    with open(embeddings_file, "wb") as f:
        pickle.dump(tyrion_embeddings, f)

# Функция поиска ответа
def get_tyrion_response(user_input):
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, tyrion_embeddings)
    best_match_idx = np.argmax(similarities)
    return tyrion_lines[best_match_idx]
