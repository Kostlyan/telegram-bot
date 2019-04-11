import telepot
import time
import urllib
import urllib.request
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import schedule
import subprocess
import os

sns.set(style="darkgrid")
#===========================================================================================================
df_finam_param = pd.read_csv('param_finam.csv', index_col=False)
#print(df_finam_param.head())

#===========================стрим данных с финама================================================================================

e = '.csv'
market = '1'

e = '.csv'
p = '7'
yf = str(datetime.datetime.now().year)
yt = str(datetime.datetime.now().year)
month_start = '01'
day_start = '02'
month_end = str(datetime.datetime.now().month)
day_end = str(datetime.datetime.now().day)
print(day_end)
dtf = '1'
tmf = '1'
MSOR = '0'
mstimever = '0'
sep = '1'
sep2 = '1'
datf = '1'
at = '0'

year_start = yf[2:]
year_end = yt[2:]
mf = (int(month_start.replace('0', ''))) - 1
mt = (int(month_end)) - 1
df = (int(day_start.replace('0', ''))) - 1
dt = (int(day_end))

def quotes(code, year_start, month_start, day_start, year_end, month_end, day_end, e, market, em, df, mf, yf, dt, mt,
           yt, p, dtf, tmf, MSOR, mstimever, sep, sep2, datf, at):
    page = urllib.request.urlopen(
        'http://export.finam.ru/' + str(code) + '_' + str(year_start) + str(month_start) + str(day_start) + '_' + str(
            year_end) + str(month_end) + str(day_end) + str(e) + '?market=' + str(market) + '&em=' + str(
            em) + '&code=' + str(code) + '&apply=0&df=' + str(df) + '&mf=' + str(mf) + '&yf=' + str(
            yf) + '&from=' + str(day_start) + '.' + str(month_start) + '.' + str(yf) + '&dt=' + str(dt) + '&mt=' + str(
            mt) + '&yt=' + str(yt) + '&to=' + str(day_end) + '.' + str(month_end) + '.' + str(yt) + '&p=' + str(
            p) + '&f=' + str(code) + '_' + str(year_start) + str(month_start) + str(day_start) + '_' + str(
            year_end) + str(month_end) + str(day_end) + '&e=' + str(e) + '&cn=' + str(code) + '&dtf=' + str(
            dtf) + '&tmf=' + str(tmf) + '&MSOR=' + str(MSOR) + '&mstimever=' + str(mstimever) + '&sep=' + str(
            sep) + '&sep2=' + str(sep2) + '&datf=' + str(datf) + '&at=' + str(at))
    global df1
    dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d %H%M%S')
    df1 = pd.read_csv(page, index_col=False, header=None, sep=',', parse_dates={'datetime': [2,3]}, date_parser=dateparse)
    df1_old = pd.read_csv(code+'.csv')
    df1.to_csv(code + '.csv')
    df1 = pd.read_csv(code + '.csv')
    len_diff = len(df1) - len(df1_old)
    if len_diff >0:
        df1_part = df1[-len_diff:]
        df1 = df1.drop(df1.index)
        df1= df1.append(df1_old, ignore_index=True)
        df1 = df1.append(df1_part, ignore_index=True)
    #os.remove(code+'.csv')
        df1.to_csv(code+'.csv')
    os.chmod(code+'.csv', 0o777)
    #df = pd.read_csv(page)
    ticker = df1['0'][-1:].values[0]
    open = df1['4'][-1:].values[0]
    high = df1['5'][-1:].values[0]
    low = df1['6'][-1:].values[0]
    global close
    close = df1['7']
    global closelast
    closelast = close[-1:]
    global prevcl
    prevcl = close[-2:-1].values[0]
    closelast = closelast.values[0]
    global perc_chg
    perc_chg = (closelast/prevcl -1)*100
    perc_chg = str(round(perc_chg, 2))+"%"
    #print(df1[-5:])
    return ticker, open, high, low, closelast, perc_chg


#===============================бот в телеге============================================================================

bot = telepot.Bot('') # ваш access key для бота
bot.deleteWebhook()
global idtable
idtable = pd.DataFrame()
idtable = idtable.append([{'chatid':164710435}])
print(idtable)
def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    print(content_type, chat_type, chat_id)
    global idtable
    global list1
    list1 = idtable['chatid'].tolist()
    if chat_id not in list1:
        idtable = idtable.append([{'chatid':chat_id}])

    print(idtable)
    if content_type == 'text':
        if any(df_finam_param.searchcode == msg["text"]):
            global code
            code = df_finam_param['code'][df_finam_param.loc[df_finam_param['searchcode']==msg["text"]].index].values[0]
            global em
            em = df_finam_param['em'][df_finam_param.loc[df_finam_param['searchcode']==msg["text"]].index].values[0]
            quote1 = quotes(code, year_start, month_start, day_start, year_end, month_end, day_end, e, market, em, df,
                            mf, yf, dt, mt, yt, p, dtf, tmf, MSOR, mstimever, sep, sep2, datf, at)
            plt.figure()  # график предсказанных значений и реальных
            plt.title('Last 45(1H) Close prices of '+ code)
            plt.xticks([])
            plt.xlabel('last 45(1H) periods')
            plt.ylabel('Price')
            plt.plot(df1.index[-45:], df1['7'][-45:])
            plt.savefig('plotclose.png')  # save the figure to file

            #====================прогнозы по запросу в телеге============

            if code == 'SBER':
                subprocess.call(['python', 'forecast_SBER.py'])
                forecast_df = pd.read_csv('Sber_Forecast.csv')
                bot.sendMessage(chat_id, "Не знаю зачем тебе оно надо, но я в чужие дела не лезу. Держи инфу по последней свече " + str(quote1)+'\n \n SBER Forecast on of this hour(used data of previous hour and do NOT take into account the current market data): '+ str(forecast_df['y_pred'][-1:].values[0]))
                bot.sendPhoto(chat_id, open('plotclose.png', 'rb'))
            elif code == 'LKOH':
                subprocess.call(['python', 'forecast_LKOH.py'])
                forecast_df = pd.read_csv('LKOH_Forecast.csv')
                bot.sendMessage(chat_id,
                                "Не знаю зачем тебе оно надо, но я в чужие дела не лезу. Держи инфу по последней свече " + str(
                                    quote1) + '\n \n LKOH Forecast on of this hour(used data of previous hour and do NOT take into account the current market data): ' + str(
                                    forecast_df['y_pred'][-1:].values[0]))
                bot.sendPhoto(chat_id, open('plotclose.png', 'rb'))
            elif code == 'USD000UTSTOM':
                subprocess.call(['python', 'forecast_USD.py'])
                forecast_df = pd.read_csv('USD_Forecast.csv')
                bot.sendMessage(chat_id,
                                "Не знаю зачем тебе оно надо, но я в чужие дела не лезу. Держи инфу по последней свече " + str(
                                    quote1) + '\n \n USD/RUB Forecast on of this hour(used data of previous hour and do NOT take into account the current market data): ' + str(
                                    forecast_df['y_pred'][-1:].values[0]))
                bot.sendPhoto(chat_id, open('plotclose.png', 'rb'))
            elif code == 'IRAO':
                subprocess.call(['python', 'forecast_IRAO.py'])
                forecast_df = pd.read_csv('IRAO_Forecast.csv')
                bot.sendMessage(chat_id,
                                "Не знаю зачем тебе оно надо, но я в чужие дела не лезу. Держи инфу по последней свече " + str(
                                    quote1) + '\n \n USD/RUB Forecast on of this hour(used data of previous hour and do NOT take into account the current market data): ' + str(
                                    forecast_df['y_pred'][-1:].values[0]))
                bot.sendPhoto(chat_id, open('plotclose.png', 'rb'))
            else:
                bot.sendMessage(chat_id, "Не знаю зачем тебе оно надо, но я в чужие дела не лезу. Держи инфу по последней свече " + str(quote1))
                bot.sendPhoto(chat_id, open('plotclose.png', 'rb'))

        elif msg["text"] == 'shed':
            sectable = pd.DataFrame(columns=('Ticker', 'Last close', '% change vs prevcl'))
            for i in df_finam_param.code:
                code = i
                em = df_finam_param['em'][df_finam_param.loc[df_finam_param['code'] == code].index].values[0]
                quote1 = quotes(code, year_start, month_start, day_start, year_end, month_end, day_end, e, market, em,
                                df, mf, yf, dt, mt, yt, p, dtf, tmf, MSOR, mstimever, sep, sep2, datf, at)
                sectable = sectable.append([{'Ticker': code, 'Last close': round(close[-2:-1].values[0], 2),
                                             '% change vs prevcl': str(
                                                 round(((close[-2:-1].values[0]) / (close[-3:-2].values[0]) - 1) * 100,
                                                       2)) + '%'}], ignore_index=True)

            subprocess.call(['python', 'forecast_trade_with_comission_SBER.py'])  # запуск прогнозного скрипта SBER
            subprocess.call(['python', 'forecast_trade_with_comission_LKOH.py'])  # запуск прогнозного скрипта LKOH
            subprocess.call(['python', 'forecast_trade_with_comission_IRAO.py'])  # запуск прогнозного скрипта IRAO

            Sber_Forecast_Test = pd.read_csv('Sber_Forecast_Test.csv')
            # print(Sber_Forecast_Test.head())
            LKOH_Forecast_Test = pd.read_csv('LKOH_Forecast_Test.csv')
            IRAO_Forecast_Test = pd.read_csv('IRAO_Forecast_Test.csv')
            virtual_trades = pd.DataFrame(columns=('Ticker', 'september profit/loss'))
            virtual_trades = virtual_trades.append([{'Ticker': Sber_Forecast_Test['0'][-1:].values[0],
                                                     'september profit/loss': round((round(
                                                         Sber_Forecast_Test['profitloss'][-1:], 2) - round(
                                                         Sber_Forecast_Test['profitloss'][1643], 2)).values[0], 2)}],
                                                   ignore_index=True)
            virtual_trades = virtual_trades.append([{'Ticker': LKOH_Forecast_Test['0'][-1:].values[0],
                                                     'september profit/loss': round(
                                                         (LKOH_Forecast_Test['profitloss'][-1:] -
                                                          LKOH_Forecast_Test['profitloss'][
                                                              1643]).values[0], 2)}], ignore_index=True)
            virtual_trades = virtual_trades.append([{'Ticker': IRAO_Forecast_Test['0'][-1:].values[0],
                                                     'september profit/loss': round(
                                                         (IRAO_Forecast_Test['profitloss'][-1:] -
                                                          IRAO_Forecast_Test['profitloss'][
                                                              1643]).values[0], 2)}], ignore_index=True)

            for counter in idtable.chatid:
                print(counter)
                bot.sendMessage(counter, str(sectable) + '\n \n' + str(virtual_trades))
        else:
            #===========ответ бота на что угодно, что не запрогано===========================
            bot.sendMessage(chat_id, "Не понимаю твои письмена, умник. Можешь попробовать ввести тикер из списка ниже или сам чекай на бирже. " + str(list(df_finam_param['searchcode'][:])))

#============================================отправка сообщений по расписанию======================================
def schedule_msg():
    sectable = pd.DataFrame(columns=('Ticker', 'Last close', '% change vs prevcl'))
    for i in df_finam_param.code:
        global code
        code = i
        global em
        em = df_finam_param['em'][df_finam_param.loc[df_finam_param['code'] == code].index].values[0]
        quote1 = quotes(code, year_start, month_start, day_start, year_end, month_end, day_end, e, market, em, df, mf, yf, dt, mt, yt, p, dtf, tmf, MSOR, mstimever, sep, sep2, datf, at)
        sectable = sectable.append([{'Ticker':code, 'Last close': round(close[-2:-1].values[0], 2), '% change vs prevcl': str(round(((close[-2:-1].values[0])/(close[-3:-2].values[0]) -1)*100,2))+'%'}], ignore_index=True)

    subprocess.call(['python', 'forecast_trade_with_comission_SBER.py'])         #запуск прогнозного скрипта SBER
    subprocess.call(['python', 'forecast_trade_with_comission_LKOH.py'])    #запуск прогнозного скрипта LKOH
    subprocess.call(['python', 'forecast_trade_with_comission_IRAO.py'])    #запуск прогнозного скрипта IRAO

    Sber_Forecast_Test = pd.read_csv('Sber_Forecast_Test.csv')
    # print(Sber_Forecast_Test.head())
    LKOH_Forecast_Test = pd.read_csv('LKOH_Forecast_Test.csv')
    IRAO_Forecast_Test = pd.read_csv('IRAO_Forecast_Test.csv')
    virtual_trades = pd.DataFrame(columns=('Ticker', 'september profit/loss'))
    virtual_trades = virtual_trades.append([{'Ticker': Sber_Forecast_Test['0'][-1:].values[0],
                                             'september profit/loss': round((round(
                                                 Sber_Forecast_Test['profitloss'][-1:], 2) - round(
                                                 Sber_Forecast_Test['profitloss'][1652], 2)).values[0], 2)}], ignore_index=True)
    virtual_trades = virtual_trades.append([{'Ticker': LKOH_Forecast_Test['0'][-1:].values[0],
                                             'september profit/loss': round((LKOH_Forecast_Test['profitloss'][-1:] -
                                                                             LKOH_Forecast_Test['profitloss'][
                                                                                 1652]).values[0], 2)}], ignore_index=True)
    virtual_trades = virtual_trades.append([{'Ticker': IRAO_Forecast_Test['0'][-1:].values[0],
                                             'september profit/loss': round((IRAO_Forecast_Test['profitloss'][-1:] -
                                                                             IRAO_Forecast_Test['profitloss'][
                                                                                 1652]).values[0], 2)}], ignore_index=True)

    for counter in idtable.chatid:
        print(counter)
        bot.sendMessage(counter, str(sectable)+'\n \n'+str(virtual_trades))

schedule.every().day.at("10:01").do(schedule_msg)
schedule.every().day.at("11:01").do(schedule_msg)
schedule.every().day.at("12:01").do(schedule_msg)
schedule.every().day.at("13:01").do(schedule_msg)
schedule.every().day.at("14:01").do(schedule_msg)
schedule.every().day.at("15:01").do(schedule_msg)
schedule.every().day.at("16:01").do(schedule_msg)
schedule.every().day.at("17:01").do(schedule_msg)
schedule.every().day.at("18:01").do(schedule_msg)
bot.message_loop(handle)

print ('Listening ...')


# Keep the program running.
while 1:
    schedule.run_pending()
    time.sleep(10)

