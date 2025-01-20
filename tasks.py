from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from celery import Celery
import pandas as pd

app = Celery('tasks', broker='amqp://guest:guest@localhost//', backend="redis://")

priors = [218334 / 253680, 35346 / 253680]
model = GaussianNB(priors=priors)

y = pd.read_csv('dados/y_diabetes.csv', header=None)

@app.task
def calcularGaussianNB(data):
    data = pd.DataFrame(data).transpose()
    x_treino, x_teste, y_treino, y_teste = train_test_split(data, y, test_size=0.2, random_state=42)

    model.fit(x_treino, y_treino)
    y_pred = model.predict(x_teste)
    recall = recall_score(y_teste, y_pred, average=None)
    print('Recall: ', recall, '\n')
    return recall

