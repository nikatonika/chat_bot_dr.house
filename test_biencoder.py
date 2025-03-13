from app.chatbot import biencoder, response_embeddings, responses
import torch

def biencoder_search(query, top_k=5):
    query_embedding = biencoder.encode(query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_embedding, response_embeddings)
    top_results = torch.topk(cos_scores, k=top_k)
    return [responses[idx] for idx in top_results.indices]

query = "I think I'm dying!"
candidates = biencoder_search(query, top_k=5)
print("Candidates found by biencoder:")
for idx, candidate in enumerate(candidates, 1):
    print(f"{idx}. {candidate}")
