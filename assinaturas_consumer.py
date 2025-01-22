from tasks import calcularGaussianNB
import pandas as pd 
from celery import group

total_linhas = 100000
lista_assinaturas = []

df_possibilidades = pd.read_csv('dados/matriz_possibilidades.csv')

for linha in range(total_linhas):
    print(linha)
    possibilidade = (df_possibilidades.iloc[linha]).tolist()
    lista_assinaturas.append(calcularGaussianNB.s(possibilidade))

grupo_tarefas = group(lista_assinaturas)
resultados = grupo_tarefas.apply_async()

for cont, resultado in enumerate(resultados):
    print(f'Resultado da task {cont}: ', resultado.get())