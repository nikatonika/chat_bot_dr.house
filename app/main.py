from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
from app.chatbot import get_house_response_fast  # Оптимизированный инференс с заглушками

app = FastAPI()

# Подключение статических файлов (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Подключение шаблонов (HTML)
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    text: str

@app.get("/")
async def read_root(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: QueryRequest):
    """Обработчик чата"""
    user_input = request.text.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Запрос не должен быть пустым")

    # Передаем в виде списка
    response = get_house_response_fast([user_input])

    # Проверяем, что ответ не пустой
    response_text = response[0] if response and isinstance(response, list) else "I don't know what to say."

    return {"response": response_text}
