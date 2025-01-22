from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from celery import Celery
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List

class ModeloLista(BaseModel):
    combinacao: List[int]

app = Celery('tasks', broker='amqp://guest:guest@localhost//', backend="redis://")

priors = [218334 / 253680, 35346 / 253680]
model = GaussianNB(priors=priors)

diabetes_binario = pd.read_csv('dados/data_tratado.csv')
diabetes_padrao = pd.read_csv('dados/diabetes_binario.csv')
y = diabetes_padrao.pop('Diabetes_binary')

@app.task
def calcularGaussianNB(vetor: list[int]):

    diabetes_combinacao = []
    for posicao, elemento in enumerate(vetor):
        if elemento == 1:
            diabetes_combinacao.append(diabetes_binario[f'{posicao}'])

    data = pd.DataFrame(diabetes_combinacao).transpose()
    x_treino, x_teste, y_treino, y_teste = train_test_split(data, y, test_size=0.2)

    model.fit(x_treino, y_treino)
    y_pred = model.predict(x_teste)
    recall = recall_score(y_teste, y_pred, average=None)
    recall_util = float(recall[1])

    return recall_util

