import telepot
import time
import urllib3
import urllib
import urllib.request
import pandas as pd
import datetime
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import schedule
import subprocess
import os
import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.utils import get_random_id
from vk_api import VkUpload

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
mt = (int(month_end.replace('0', ''))) - 1
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
    os.remove(code+'.csv')
    df1.to_csv(code+'.csv')
    #df = pd.read_csv(page)
    ticker = df1[0][-1:].values[0]
    open = df1[4][-1:].values[0]
    high = df1[5][-1:].values[0]
    low = df1[6][-1:].values[0]
    global close
    close = df1[7]
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

vk_session = vk_api.VkApi(token='')
longpoll = VkBotLongPoll(vk_session, '178962083')
vk = vk_session.get_api()

def main():
    for event in longpoll.listen():
        if any(df_finam_param.searchcode == event.obj.text):
            global code
            code = df_finam_param['code'][df_finam_param.loc[df_finam_param['searchcode'] == event.obj.text].index].values[
                0]
            global em
            em = df_finam_param['em'][df_finam_param.loc[df_finam_param['searchcode'] == event.obj.text].index].values[0]
            quote1 = quotes(code, year_start, month_start, day_start, year_end, month_end, day_end, e, market, em, df,
                            mf, yf, dt, mt, yt, p, dtf, tmf, MSOR, mstimever, sep, sep2, datf, at)
            plt.figure()  # график предсказанных значений и реальных
            plt.title('Last 45(1H) Close prices of ' + code)
            plt.xticks([])
            plt.xlabel('last 45(1H) periods')
            plt.ylabel('Price')
            plt.plot(df1.index[-45:], df1[7][-45:])
            plt.savefig('plotclose.png')  # save the figure to file

            # ====================прогнозы по запросу в телеге============

            if code == 'SBER':
                subprocess.call(['python', 'forecast_SBER.py'])
                forecast_df = pd.read_csv('Sber_Forecast.csv')
                photos = ['plotclose.png']
                upload = VkUpload(vk_session)
                photo_list = upload.photo_messages(photos)
                vk.messages.send(peer_id='2000000001',
                random_id=get_random_id(),
                                attachment=','.join('photo{owner_id}_{id}'.format(**item) for item in photo_list),
                                message="Не знаю зачем тебе оно надо, но я в чужие дела не лезу. Держи инфу по последней свече " + str(
                                    quote1) + '\n \n SBER Forecast on of this hour(used data of previous hour and do NOT take into account the current market data): ' + str(
                                    forecast_df['y_pred'][-1:].values[0]))

            elif code == 'LKOH':
                subprocess.call(['python', 'forecast_LKOH.py'])
                forecast_df = pd.read_csv('LKOH_Forecast.csv')
                photos = ['plotclose.png']
                upload = VkUpload(vk_session)
                photo_list = upload.photo_messages(photos)
                vk.messages.send(peer_id='2000000001',
                random_id=get_random_id(),
                                attachment=','.join('photo{owner_id}_{id}'.format(**item) for item in photo_list),
                                message="Не знаю зачем тебе оно надо, но я в чужие дела не лезу. Держи инфу по последней свече " + str(
                                    quote1) + '\n \n LKOH Forecast on of this hour(used data of previous hour and do NOT take into account the current market data): ' + str(
                                    forecast_df['y_pred'][-1:].values[0]))

            elif code == 'USD000UTSTOM':
                subprocess.call(['python', 'forecast_USD.py'])
                forecast_df = pd.read_csv('USD_Forecast.csv')
                photos = ['plotclose.png']
                upload = VkUpload(vk_session)
                photo_list = upload.photo_messages(photos)
                vk.messages.send(peer_id='2000000001',
                random_id=get_random_id(),
                                attachment=','.join('photo{owner_id}_{id}'.format(**item) for item in photo_list),
                                message="Не знаю зачем тебе оно надо, но я в чужие дела не лезу. Держи инфу по последней свече " + str(
                                    quote1) + '\n \n USD/RUB Forecast on of this hour(used data of previous hour and do NOT take into account the current market data): ' + str(
                                    forecast_df['y_pred'][-1:].values[0]))

            elif code == 'IRAO':
                subprocess.call(['python', 'forecast_IRAO.py'])
                forecast_df = pd.read_csv('IRAO_Forecast.csv')
                photos = ['plotclose.png']
                upload = VkUpload(vk_session)
                photo_list = upload.photo_messages(photos)
                vk.messages.send(peer_id='2000000001',
                random_id=get_random_id(),
                                attachment=','.join('photo{owner_id}_{id}'.format(**item) for item in photo_list),
                                message="Не знаю зачем тебе оно надо, но я в чужие дела не лезу. Держи инфу по последней свече " + str(
                                    quote1) + '\n \n USD/RUB Forecast on of this hour(used data of previous hour and do NOT take into account the current market data): ' + str(
                                    forecast_df['y_pred'][-1:].values[0]))

            else:
                photos = ['plotclose.png']
                upload = VkUpload(vk_session)
                photo_list = upload.photo_messages(photos)
                vk.messages.send(peer_id='2000000001',
                random_id=get_random_id(),
                                attachment=','.join('photo{owner_id}_{id}'.format(**item) for item in photo_list),
                                message="Не знаю зачем тебе оно надо, но я в чужие дела не лезу. Держи инфу по последней свече " + str(
                                    quote1))

        else:
            # ===========ответ бота на что угодно, что не запрогано===========================
            vk.messages.send(peer_id='2000000001',
                random_id=get_random_id(),
                            message="Не понимаю твои письмена, умник. Можешь попробовать ввести тикер из списка ниже или валить из этого чата нахер. " + str(
                                list(df_finam_param['searchcode'][:])))

def schedule_msg():
    vk_session = vk_api.VkApi(
    token='50e4e1517114db9406c4e708b09138391c6a8ae570eece83e30a24725f66f83f7d0b400a4cee42d1b5f4a')
    longpoll = VkBotLongPoll(vk_session, '178962083')
    vk = vk_session.get_api()
    print('test')
    vk.messages.send(  # Отправляем сообщение
        peer_id='2000000001',
        random_id=get_random_id(),
        message=str('test message')+'\n \n'+str('!')
    )


schedule.every().day.at("10:02").do(schedule_msg)
schedule.every().day.at("11:02").do(schedule_msg)
schedule.every().day.at("12:02").do(schedule_msg)
schedule.every().day.at("13:02").do(schedule_msg)
schedule.every().day.at("14:02").do(schedule_msg)
schedule.every().day.at("15:02").do(schedule_msg)
schedule.every().day.at("16:02").do(schedule_msg)
schedule.every().day.at("17:02").do(schedule_msg)
schedule.every().day.at("18:02").do(schedule_msg)
schedule.every().day.at("22:59").do(schedule_msg)
#bot.message_loop(handle)

print ('Listening ...')


# Keep the program running.
while 1:
    schedule.run_pending()
    time.sleep(10)
    main()
