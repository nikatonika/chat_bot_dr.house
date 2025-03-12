from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.chatbot import get_house_response 

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

    response = get_house_response(user_input)
    return {"response": response}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
