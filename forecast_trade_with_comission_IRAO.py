import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv, DataFrame, Series
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle
import math
import datetime
import os

scaler = MinMaxScaler()

df1 = read_csv('IRAO.csv')  # заливаем csv файл с коировками, скаченными с finam
ticker_code = 'IRAO'
#df_main = read_csv('IRAO_Forecast_Test.csv')

a1 = len(df1.index)-100  # переменная, которая задает количество прогнозов, которые мыхотим получить, можно так же

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

nn = MLPClassifier(hidden_layer_sizes=(63, 86), max_iter=20000, alpha=0.00001, random_state=0)  # задаем модель

pickle_in = open('IRAO_1.pickle','rb')
nn = pickle.load(pickle_in)

y_pred = nn.predict(X_test)  # прогноз, который нам выдает по тестовой выборке

date1 = df1['datetime']
date1 = date1.values.reshape(-1, 1)  # для построения графика

print("Правильность на тестовом наборе: {:.2f}".format(nn.score(X_test, y_test)))

df2 = df1[-a1:-1]  # создаем новый датафрейм для удобства оценки предсказанных результатов
df2 = df2.reset_index()  # сбрасываем индексы старого датафрейма
df2['y_pred'] = pd.Series(y_pred, index=df2.index)  # добавляем предсказанные результаты новой колонкой в датафрейм

print(df2[['y_pred', 'label']])

#df2.to_csv('Sber_Forecast.csv')
deals = pd.DataFrame(columns=('Ticker', 'trade_datetime', 'type', 'volume', 'PnL'))

Capital = []
Deals = []
quantity = []
longclmn = []
shortclmn = []
profitloss = []
commision = []
comm_perc = 0.00035  #проценты в долях
price_fail = 0.0001 #проценты в долях
profitcount = 10000
quant12 = 0
ii = -1

for row in df2['y_pred']:
    ii = ii + 1
    #if ii-1<=0:
        #quant12 = 0
    #else:
        #quant12 =df_main['quantity'][ii-1]

    if quant12 == 0:
        if row == 'up':
            #если текущая позиция 0 и тренд вверх => покупаем
            close = df2['nextclose'][ii]
            quant12 = math.floor(2500)
            quantity.append(quant12)
            Deals.append((1+price_fail)*close*quant12)
            profitcount = profitcount - close*quant12
            Capital.append(profitcount)
            longclmn.append(-(1+price_fail)*close*quant12)
            comm = math.fabs((1+price_fail)*close*quant12)* comm_perc
            commision.append(comm)
            shortclmn.append(0)
            profitloss.append(profitcount+(1+price_fail)*close*quant12-10000-comm)
            deals = deals.append([{'Ticker': ticker_code, 'trade_datetime': df2['datetime'][ii], 'type': 'long', 'volume': -(1+price_fail)*close*quant12, 'PnL': 0}])

        else:
            # если текущая позиция 0 и тренд вниз => продаем
            close = df2['nextclose'][ii]
            quant12 = -math.floor(2500)
            quantity.append(quant12)
            Deals.append((1-price_fail)*close * quant12)
            profitcount = profitcount - (1-price_fail)*close * quant12
            Capital.append(profitcount)
            longclmn.append(0)
            shortclmn.append((1-price_fail)*close*quant12)
            comm = math.fabs((1-price_fail)*close*quant12) * comm_perc
            commision.append(comm)
            profitloss.append(profitcount+(1-price_fail)*close*quant12-10000-comm)
            deals = deals.append([{'Ticker': ticker_code, 'trade_datetime': df2['datetime'][ii], 'type': 'short', 'volume': -(1 - price_fail) * close * quant12, 'PnL': 0}])
    elif quant12 > 0:
        if row == 'up':
            # если текущая позиция лонг и тренд вверх => холдим
            Deals.append(0)
            Capital.append(profitcount)
            quantity.append(quant12)
            longclmn.append(0)
            shortclmn.append(0)
            comm = 0
            commision.append(comm)
            profitloss.append(profitcount+(1+price_fail)*close*quant12-10000)
        else:
            # если текущая позиция лонг и тренд вниз => закрываем лонг и открываем шорт
            close = df2['nextclose'][ii]
            i = (1-price_fail)*close * math.fabs(quant12)
            longclmn.append((1-price_fail)*close * quant12)
            deals = deals.append([{'Ticker': ticker_code, 'trade_datetime': df2['datetime'][ii], 'type': 'long', 'volume': (1 - price_fail) * close * quant12, 'PnL': (float(deals['volume'][-1:].values)+(1 - price_fail) * close * quant12)}])
            comm = math.fabs((1-price_fail)*close * quant12) * comm_perc
            profitcount = profitcount + (1-price_fail)*close * math.fabs(quant12)
            quant12 = -math.floor(2500)
            Deals.append((1-price_fail)*close * quant12 - i)
            profitcount = profitcount - (1-price_fail)*close * quant12
            quantity.append(quant12)
            Capital.append(profitcount)
            shortclmn.append((1-price_fail)*close*quant12)
            deals = deals.append([{'Ticker': ticker_code, 'trade_datetime': df2['datetime'][ii], 'type': 'short', 'volume': -(1-price_fail)*close*quant12,
                                   'PnL': 0}])
            comm2 = math.fabs((1-price_fail)*close*quant12) * comm_perc
            commision.append(comm+comm2)
            profitloss.append(profitcount+(1-price_fail)*close*quant12-10000-comm-comm2)
    elif quant12 < 0:
        if row == 'down':
            # если текущая позиция шорт и тренд вниз => холдим
            Deals.append(0)
            Capital.append(profitcount)
            quantity.append(quant12)
            longclmn.append(0)
            shortclmn.append(0)
            comm = 0
            commision.append(comm)
            profitloss.append(profitcount + (1-price_fail)*close * (quant12) - 10000)
        else:
            # если текущая позиция шорт и тренд вверх => закрываем шорт и открываем лонг
            close = df2['nextclose'][ii]
            i = (1+price_fail)*close * math.fabs(quant12)
            shortclmn.append(i)
            deals = deals.append([{'Ticker': ticker_code, 'trade_datetime': df2['datetime'][ii], 'type': 'short', 'volume': -i,
                                   'PnL': (float(deals['volume'][-1:].values)-i)}])
            comm = math.fabs(i) * comm_perc
            profitcount = profitcount - (1+price_fail)*close * math.fabs(quant12)
            quant12 = math.floor(2500)
            Deals.append((1+price_fail)*close * quant12 + i)
            profitcount = profitcount - (1+price_fail)*close * quant12
            Capital.append(profitcount)
            quantity.append(quant12)
            longclmn.append(-(1+price_fail)*close*quant12)
            deals = deals.append([{'Ticker': ticker_code, 'trade_datetime': df2['datetime'][ii], 'type': 'long', 'volume': -(1+price_fail)*close*quant12,
                                   'PnL': 0}])
            comm2 = math.fabs((1+price_fail)*close*quant12) * comm_perc
            commision.append(comm + comm2)
            profitloss.append(profitcount+(1+price_fail)*close*quant12 - 10000-comm-comm2)

df2['Capital'] = pd.Series(Capital)
df2['Deals'] = pd.Series(Deals)
df2['quantity'] = pd.Series(quantity)
df2['long'] = pd.Series(longclmn)
df2['short'] = pd.Series(shortclmn)
df2['profitloss'] = pd.Series(profitloss)
df2['commision'] = pd.Series(commision)


deals.to_csv('deals_irao.csv')
df2.to_csv('IRAO_Forecast_Test.csv')
now = datetime.datetime.now()
#df2.to_csv('/opt/python/archive/IRAO_Forecast_Test'+str(now.year)+str(now.month)+str(now.day)+str(now.hour)+str(now.minute)+str(now.second)+'.csv')
