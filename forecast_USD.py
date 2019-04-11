import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv, DataFrame, Series
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle

scaler = MinMaxScaler()

df1 = read_csv('USD000UTSTOM.csv')  # заливаем csv файл с коировками, скаченными с finam

a1 = 10  # переменная, которая задает количество прогнозов, которые мыхотим получить, можно так же

prst_op_cl = df1['4'] / df1['7'] - 1
prst_hi_cl = df1['5'] / df1['7'] - 1
prst_lo_cl = df1['6'] / df1['7'] - 1
df1['prst_lo_cl'] = pd.Series(prst_lo_cl)
df1['prst_hi_cl'] = pd.Series(prst_hi_cl)
df1['prst_op_cl'] = pd.Series(prst_op_cl)

prevclose = df1['7']  # n-1
prevclose = pd.Series(prevclose, dtype=float)
prevclose = prevclose[13:-1]
prevclose = prevclose.reset_index()
prevclose = prevclose['7']

nextclose = df1['7']
nextclose = pd.Series(nextclose, dtype=float)
nextclose = nextclose[15:]
nextclose = nextclose.reset_index()
nextclose = nextclose['7']

df1 = df1[14:]
df1 = df1.reset_index()
df1['prevclose'] = pd.Series(prevclose)
df1['nextclose'] = pd.Series(nextclose)

prst_prevcl_cl = df1['prevclose'] / df1['7'] - 1
prst_cl_nextcl = df1['nextclose'] / df1['7'] - 1

df1['prst_prevcl_cl'] = pd.Series(prst_prevcl_cl)
df1['prst_cl_nextcl'] = pd.Series(prst_cl_nextcl)

label = []
for row in df1['prst_cl_nextcl']:
    if row > 0:
        label.append('up')
    else:
        label.append('down')

df1['label'] = pd.Series(label)

train_df1 = df1[
    ['4', '5', '6', 'nextclose', 'prevclose', 'prst_prevcl_cl', 'prst_lo_cl', 'prst_hi_cl', 'prst_op_cl']]

X_train = train_df1[:-a1]  # обучающая выборка

X_test = train_df1[-a1:-1]  # тестовая выборка

y_train = df1['label'][:-a1]  # цель обучающей выборки

y_test = df1['label'][-a1:-1]  # реальные данные соответствующие тестовой выборке, цель, которую нужно в идеале
# достич нашей прогнозной моделью

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

y_test = y_test.values.reshape(-1, 1)

nn = MLPClassifier(hidden_layer_sizes=(20, 87), max_iter=20000, alpha=0.00001, random_state=0)  # задаем модель

pickle_in = open('USD_1.pickle','rb')
nn = pickle.load(pickle_in)

y_pred = nn.predict(X_test)  # прогноз, который нам выдает по тестовой выборке

date1 = df1['datetime']
date1 = date1.values.reshape(-1, 1)  # для построения графика

print("Правильность на тестовом наборе: {:.2f}".format(nn.score(X_test, y_test)))

df2 = df1[-a1:-1]  # создаем новый датафрейм для удобства оценки предсказанных результатов
df2 = df2.reset_index()  # сбрасываем индексы старого датафрейма
df2['y_pred'] = pd.Series(y_pred, index=df2.index)  # добавляем предсказанные результаты новой колонкой в датафрейм

print(df2[['y_pred', 'label']])

df2.to_csv('USD_Forecast.csv')
