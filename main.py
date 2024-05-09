from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str

class ApiParams(BaseModel):
   search_topic: str
   question: str

# Функция подгрузки данных модели ответов
def load_answer_model():
   model_pipeline = pipeline(task='question-answering', model='deepset/roberta-base-squad2')
   return model_pipeline

def load_classifier_model():
    classifier = pipeline("sentiment-analysis")
    return classifier


model = load_answer_model()
classifier = load_classifier_model()
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]['label']

@app.post("/answer/")
def answer(params: ApiParams):
   # Ответ на вопрос
   result = model(params.question, params.search_topic)
   return result