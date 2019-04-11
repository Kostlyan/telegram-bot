import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv, DataFrame, Series
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

scaler = MinMaxScaler()

df1 = read_csv('SBER.csv')  # заливаем csv файл с коировками, скаченными с finam

a1 = 10  # переменная, которая задает количество прогнозов, которые мыхотим получить, можно так же
# задать как a1= int(len(df1)*0.1), если хотим проверить нашу модель на 10% от выборки

#print("Ключи sber: \n{}".format(df1.keys()))  # вспомогательная строка, показывает какие колонки у нас есть(только названия),
# можно удалить

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
# scaler.fit(y_test)
# y_test_scaled = scaler.transform(y_test)

nn = MLPClassifier(hidden_layer_sizes=(50, 10), max_iter=20000, alpha=0.5, random_state=0)  # задаем модель

# X_train = X_train.values.reshape(-1, 1)     #подготавливаем выборки
# y_train = y_train.values.reshape(-1, 1)
# X_test = X_test.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

#nn.fit(X_train, y_train.ravel())  # обучаем
#with open('mlpclassifier3.pickle','wb') as d:
    #pickle.dump(nn, d)
pickle_in = open('mlpclassifier3.pickle','rb')
nn = pickle.load(pickle_in)

y_pred = nn.predict(X_test)  # прогноз, который нам выдает по тестовой выборке
#print(X_test)

date1 = df1['datetime']
date1 = date1.values.reshape(-1, 1)  # для построения графика

print("Правильность на тестовом наборе: {:.2f}".format(nn.score(X_test, y_test)))

# y_pred=pd.Series(y_pred, dtype=float)              #форматируем в формат series pandas
df2 = df1[-a1:-1]  # создаем новый датафрейм для удобства оценки предсказанных результатов
df2 = df2.reset_index()  # сбрасываем индексы старого датафрейма
df2['y_pred'] = pd.Series(y_pred, index=df2.index)  # добавляем предсказанные результаты новой колонкой в датафрейм

# diff_btw_targ_pred =(df2['y_pred'] - df2['label'])      #разница предсказанных значений и реальных
# max1 = diff_btw_targ_pred[diff_btw_targ_pred>0].max()
# min1 = diff_btw_targ_pred[diff_btw_targ_pred>0].min()
# avg1 = diff_btw_targ_pred[diff_btw_targ_pred>0].mean()
# max2 = diff_btw_targ_pred[diff_btw_targ_pred<0].max()
# min2 = diff_btw_targ_pred[diff_btw_targ_pred<0].min()
# avg2 = diff_btw_targ_pred[diff_btw_targ_pred<0].mean()
# print("разница прогноза и исходных данных:\n {}".format(diff_btw_targ_pred))
# print(max1,min1,max2,min2,avg1,avg2)          #min,max,avg положительных и отрицательных разниц
print(df2[['y_pred', 'label']])
# plt.figure(figsize=(80,30))                   #график предсказанных значений и реальных
# plt.plot(date1[-a1:-1], df2['y_pred'])
# plt.plot(date1[-a1:-1], df2['label'],c='r')
# plt.figure(figsize=(80,30))
# plt.plot(diff_btw_targ_pred)                  #график разницы предсказанных значений и реальных

#df1.to_csv('df1.csv')
df2.to_csv('Sber_Forecast.csv')
