from fastapi import FastAPI  
from transformers import pipeline  
from pydantic import BaseModel  
from collections import Counter  
import re  

class Item(BaseModel):  
    text: str  

app = FastAPI()  
classifier = pipeline("sentiment-analysis")  

@app.get("/")  
def root() -> dict:  
    """Возвращает сообщение о запуске FastAPI сервиса."""  
    return {"message": "FastAPI service started!"}  

@app.get("/{text}")  
def get_params(text: str) -> dict:  
    """Возвращает результат анализа настроения для переданного текста."""  
    return classifier(text)  

@app.post("/predict/")  
def predict(item: Item) -> dict:  
    """Возвращает результат анализа настроения для текста из объекта Item."""  
    return classifier(item.text)  

@app.post("/most_common_words/")  
def most_common_words(item: Item, num: int = 5) -> dict:  
    """Возвращает наиболее часто встречающиеся слова в тексте."""  
    # Удаляем знаки препинания и разбиваем текст на слова  
    words = re.findall(r'\b\w+\b', item.text.lower())  
    # Считаем частоту слов  
    word_counts = Counter(words)  
    # Получаем наиболее распространённые слова  
    most_common = word_counts.most_common(num)  
    
    return {"most_common_words": most_common}
