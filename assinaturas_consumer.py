import pandas as pd
import numpy as np
from tasks import calcularGaussianNB
from celery import group

diabetes_binario = pd.read_csv('dados/data_tratado.csv')
matriz_possibilidades = pd.read_csv('dados/matriz_possibilidades.csv')


total_linhas = 2**21 - 1
lista_assinaturas = []

for i in range(1000):
    array_posicao = np.array(matriz_possibilidades.iloc[i])
    diabetes_posicao = []
    
    for k in range(21):
        if array_posicao[k] == 1:
            diabetes_posicao.append(diabetes_binario[f'{k}'])
    
    print(f"Processando linha {i}")
    lista_assinaturas.append(calcularGaussianNB.s(diabetes_posicao))

grupo_tarefas = group(lista_assinaturas)
resultado = grupo_tarefas.apply_async()

print(resultado.get)