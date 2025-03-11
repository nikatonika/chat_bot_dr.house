import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Проверка наличия GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используем устройство: {device}")

# Загружаем данные
file_path = "data/game-of-thrones.csv"
df = pd.read_csv(file_path)

# Фильтруем реплики Тириона
tyrion_lines = df[df["Speaker"] == "TYRION"]["Text"].dropna().tolist()

# Формируем пары реплик (вопрос-ответ)
train_data = [InputExample(texts=[tyrion_lines[i], tyrion_lines[i+1]], label=1.0) for i in range(len(tyrion_lines)-1)]

# Загружаем предобученную Sentence-BERT модель
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Настраиваем DataLoader
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
train_loss = losses.CosineSimilarityLoss(model)

# Оценка перед обучением
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(train_data[:100], name="tyrion-eval")

# Обучение модели
epochs = 5
warmup_steps = int(len(train_dataloader) * epochs * 0.1)
history = {"loss": []}

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # Обучение модели
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        output_path="fine_tuned_tyrion_model"
    )

    # Оценка модели
    # Преобразование входных данных в нужный формат (InputExample)
    features = [InputExample(texts=[tyrion_lines[i], tyrion_lines[i+1]], label=1.0) for i in range(10)]

    # Создание DataLoader для вычисления потерь
    # Кодируем входные данные (конвертируем в эмбеддинги)
    encoded_sentences_1 = model.encode([tyrion_lines[i] for i in range(10)], convert_to_tensor=True)
    encoded_sentences_2 = model.encode([tyrion_lines[i+1] for i in range(10)], convert_to_tensor=True)

    # Вычисляем ошибку потерь с `CosineSimilarityLoss`
    with torch.no_grad():
        loss_value = train_loss(encoded_sentences_1, encoded_sentences_2).item()

    print(f"Loss: {loss_value:.4f}")


    print(f"Loss: {loss_value:.4f}")


    history["loss"].append(loss_value)

# Сохранение обученной модели
model.save("fine_tuned_tyrion_model")
print("Модель сохранена!")

# 📊 График обучения
plt.figure(figsize=(10, 5))
plt.plot(history["loss"], label="Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("График потерь при обучении")
plt.grid(True)
plt.show()
