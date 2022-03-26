import mysql.connector
import urllib.parse
import pandas as pd
import numpy as np
import requests
import datetime
import requests
import catboost
import pymysql
import pathlib
import urllib
import pyodbc 
import optuna
import json
import yaml
import os
from datetime import time
from sklearn.metrics import r2_score
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score
from catboost import Pool, CatBoostRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


######### Общий раздел №№№№№№№№№№

""" Загружаем yaml файл с настройками """
with open(str(pathlib.Path(__file__).parent.absolute())+'/settings.yaml', 'r') as yaml_file:
    settings = yaml.safe_load(yaml_file)
telegram_settings = pd.DataFrame(settings['telegram'])
sql_settings = pd.DataFrame(settings['sql_db'])
pyodbc_settings = pd.DataFrame(settings['pyodbc_db'])
vc_settings = pd.DataFrame(settings['visualcrossing_api'])

""" Функция отправки уведомлений в tekegram на любое количество каналов (указать данные в yaml файле настроек) """
def telegram(text):
    msg = urllib.parse.quote(str(text))
    for channel in range(len(telegram_settings.index)):
        bot_token = str(telegram_settings.bot_token[channel])
        channel_id = str(telegram_settings.channel_id[channel])
        requests.get('https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + channel_id + '&text=' + msg)

""" Функция коннекта к базе Mysql (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)"""
def connection(i):
    host_yaml = str(sql_settings.host[i])
    user_yaml = str(sql_settings.user[i])
    port_yaml = int(sql_settings.port[i])
    password_yaml = str(sql_settings.password[i])
    database_yaml = str(sql_settings.database[i])
    return pymysql.connect(host=host_yaml,user=user_yaml,port=port_yaml,password=password_yaml,database=database_yaml)

############## Раздел загрузки прогноза погоды в базу ##################

""" Уведомление о старте задания """
""" telegram("VisualCrossing: загрузка прогноза запущена") """

""" Задаем переменные (даты для прогноза и токен для получения прогноза) """
date_beg = (datetime.datetime.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
date_end = (datetime.datetime.today() + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
api_key= str(vc_settings.api_key[0])
x_test = pd.DataFrame()
""" Загрузка списка ГТП и координат из базы """
connection_geo = connection(0)
with connection_geo.cursor() as cursor:
    sql = "select gtp,lat,lng from visualcrossing.ses_gtp;"  
    cursor.execute(sql)    
    ses_dataframe=pd.DataFrame(cursor.fetchall(), columns=['gtp','lat','lng'])
    connection_geo.close()

g = 0 
for ses in range(len(ses_dataframe.index)):
    gtp = str(ses_dataframe.gtp[ses])
    lat = str(ses_dataframe.lat[ses]).replace(',','.')
    lng = str(ses_dataframe.lng[ses]).replace(',','.')
    print(gtp)
    try:
        url_response = requests.get("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/" + lat + "," + lng + "/" + date_beg + "/" + date_end + "?unitGroup=metric&key=" + api_key + "&include=fcst%2Chours&elements=datetime,sunrise,sunset,temp,dew,humidity,precip,precipprob,preciptype,snow,snowdepth,windgust,windspeed,pressure,cloudcover,visibility,solarradiation,solarenergy,uvindex,severerisk,conditions")
        url_response.raise_for_status()
        if url_response.ok:
            json_string = json.loads(url_response.text)
            dataframe_1 = pd.DataFrame(data=json_string['days'])
            g+=1
            print("прогноз погоды загружен")
        else:
            print('VisualCrossing: Ошибка запроса:' + str(url_response.status_code))
            telegram('VisualCrossing: Ошибка запроса:' + str(url_response.status_code))
            os._exit(1)
    except requests.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err.response.text}')
        telegram(f'HTTP error occurred: {http_err.response.text}')
        os._exit(1)
    except Exception as err:
        print(f'Other error occurred: {err}')
        telegram(f'Other error occurred: {err}')
        os._exit(1)

    dataframe_3 = pd.DataFrame()
    for i in range(len(dataframe_1.index)):
        dataframe_2 = pd.DataFrame(data=dataframe_1['hours'] [i])
        dataframe_2['datetime'] = dataframe_1['datetime'] [i] + ' ' + dataframe_2['datetime']
        dataframe_2['sunrise'] = dataframe_1['sunrise'] [i]
        dataframe_2['sunset'] = dataframe_1['sunset'] [i]
        dataframe_2['gtp'] = gtp
        dataframe_2['tzoffset'] = json_string['tzoffset']
        tzoffset = dataframe_2.tzoffset[i]-3
        dataframe_2['datetime_msc'] = pd.to_datetime(dataframe_2['datetime'], utc=False)- pd.DateOffset(hours=tzoffset)
        dataframe_3 = dataframe_3.append(dataframe_2, ignore_index=True)
    """ print(dataframe_3) """
    x_test = x_test.append(dataframe_3, ignore_index=True)
""" print(x_test) """

""" Уведомление о количестве загруженных прогнозов """
""" telegram("VisualCrossing: загружен прогноз для " + str(g) + " гтп") """

connection_vc = connection(0)
conn_cursor = connection_vc.cursor()

vall = ''
rows = len(x_test.index)
gtp_rows = int(round(rows/72,0))
for r in range(len(x_test.index)):
    vall= (vall+"('"
    +str(x_test.gtp[r])+"','"
    +str(x_test.datetime[r])+"','"
    +str(x_test.tzoffset[r])+"','"
    +str(x_test.datetime_msc[r])+"','"                       
    +str(datetime.datetime.now().isoformat())+"','"                       
    +str(x_test.sunrise[r])+"','"                      
    +str(x_test.sunset[r])+"','"                      
    +str(x_test.temp[r])+"','"                       
    +str(x_test.dew[r])+"','"                      
    +str(x_test.humidity[r])+"','"                       
    +str(x_test.precip[r]).replace('nan','0.0')+"','"                      
    +str(x_test.precipprob[r]).replace('nan','0.0')+"','"                      
    +str(x_test.preciptype[r]).replace("""['""",'').replace("""']""",'').replace("""', '""",",").replace(' ','None')+"','"                       
    +str(x_test.snow[r]).replace('nan','0.0')+"','"                    
    +str(x_test.snowdepth[r])+"','"                       
    +str(x_test.windgust[r])+"','"                      
    +str(x_test.windspeed[r])+"','"                       
    +str(x_test.pressure[r])+"','"                       
    +str(x_test.cloudcover[r])+"','"                      
    +str(x_test.visibility[r])+"','"                      
    +str(x_test.solarradiation[r])+"','"                       
    +str(x_test.solarenergy[r]).replace('nan','0.0').replace(' ','0.0').replace('NaN','0.0')+"','"                      
    +str(x_test.uvindex[r])+"','"                      
    +str(x_test.severerisk[r])+"','"                      
    +str(x_test.conditions[r])+"'"+'),')

vall = vall[:-1]
sql = ("INSERT INTO visualcrossing.forecast (gtp,datetime,tzoffset,datetime_msc,loadtime,sunrise,sunset,temp,dew,humidity,precip,precipprob,preciptype,snow,snowdepth,windgust,windspeed,pressure,cloudcover,visibility,solarradiation,solarenergy,uvindex,severerisk,conditions) VALUES " + vall + ";")
conn_cursor.execute(sql) 
connection_vc.commit()
connection_vc.close()

""" Уведомление о записи в БД """
telegram("VisualCrossing: записано в БД " + str(rows) + " строк (" + str(gtp_rows) + " гтп)")

################### Раздел подготовки прогноза на catboost ###################
""" Загрузка прогноза погоды из базы и подготовка датафреймов """
connection_geo = connection(0)
with connection_geo.cursor() as cursor:
    sql = "select gtp,def_power from visualcrossing.ses_gtp;"  
    cursor.execute(sql)    
    ses_dataframe=pd.DataFrame(cursor.fetchall(), columns=['gtp','def_power'])
    ses_dataframe['def_power']=ses_dataframe['def_power']*1000

    ses_dataframe=ses_dataframe[ses_dataframe['gtp'].str.contains('G', regex=False)]
    connection_geo.close()

connection_forecast = connection(0)
with connection_forecast.cursor() as cursor:
    sql = "select gtp,datetime_msc,tzoffset,loadtime,sunrise,sunset,temp,dew,humidity,precip,precipprob,preciptype,snow,snowdepth,windgust,windspeed,pressure,cloudcover,visibility,solarradiation,solarenergy,uvindex,severerisk,conditions from visualcrossing.forecast where loadtime >= CURDATE() - INTERVAL 39 DAY;"  
    cursor.execute(sql)    
    forecast_dataframe=pd.DataFrame(cursor.fetchall(), columns=['gtp','datetime_msc','tzoffset','loadtime','sunrise','sunset','temp','dew','humidity','precip','precipprob','preciptype','snow','snowdepth','windgust','windspeed','pressure','cloudcover','visibility','solarradiation','solarenergy','uvindex','severerisk','conditions'])
    connection_forecast.close()

""" Удаление дубликатов прогноза, т.к. каждый день грузит на 3 дня вперед и получается накладка """
date_beg_predict = (datetime.datetime.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
date_end_predict = (datetime.datetime.today() + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
forecast_dataframe.drop_duplicates(subset=['datetime_msc','gtp'], keep='last', inplace=True, ignore_index=False)
forecast_dataframe['hour']=forecast_dataframe['datetime_msc'].str[-8:-6].astype('float')
forecast_dataframe['cloudcover']=100-forecast_dataframe['cloudcover'].astype('float')
test_dataframe=forecast_dataframe.drop(forecast_dataframe.index[np.where(forecast_dataframe['datetime_msc'] < str(date_beg_predict))[0]])
test_dataframe.drop(forecast_dataframe.index[np.where(forecast_dataframe['datetime_msc'] > str(date_end_predict))[0]], inplace=True)
test_dataframe=test_dataframe[test_dataframe['gtp'].str.contains('G', regex=False)]
test_dataframe=test_dataframe.merge(ses_dataframe, left_on=['gtp'], right_on = ['gtp'], how='right')

forecast_dataframe.drop(forecast_dataframe.index[np.where(forecast_dataframe['datetime_msc'] > str(datetime.datetime.today()))[0]], inplace=True)
""" Сортировка датафрейма по гтп и дате """
forecast_dataframe.sort_values(['gtp','datetime_msc'], inplace=True)
forecast_dataframe['datetime_msc']=forecast_dataframe['datetime_msc'].astype('datetime64[ns]')

""" Загрузка факта выработки """
server = str(pyodbc_settings.host[0]) 
database = str(pyodbc_settings.database[0])
username = str(pyodbc_settings.user[0]) 
password = str(pyodbc_settings.password[0])
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

########!!!!!!!!!!!!!!!!!В строку ниже записать sql запрос для загрузки фактической выработки за необходимое количество предыдущих дней (обычно от 45 до 31)!!!!!!!##########
cursor.execute("SELECT fact  FROM ..... >= DATEADD(HOUR, -39 * 24, DATEDIFF(d, 0, GETDATE().....ORDER BY gtp;")
###############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
fact = cursor.fetchall()
cnxn.close()
fact=pd.DataFrame(np.array(fact), columns=['gtp','dt','fact'])
fact.drop_duplicates(subset=['gtp','dt'], keep='last', inplace=True, ignore_index=False)

forecast_dataframe=forecast_dataframe.merge(ses_dataframe, left_on=['gtp'], right_on = ['gtp'], how='right')
forecast_dataframe=forecast_dataframe.merge(fact, left_on=['gtp','datetime_msc'], right_on = ['gtp','dt'], how='right')
forecast_dataframe.dropna(subset=['datetime_msc'], inplace=True)
forecast_dataframe.drop(['dt','loadtime','sunrise','sunset'], axis='columns', inplace=True)
""" forecast_dataframe.to_excel("forecast_dataframe2.xlsx") """
""" test_dataframe.to_excel("test_dataframe2.xlsx") """

""" CatBoost """
z=forecast_dataframe.drop(forecast_dataframe.index[np.where(forecast_dataframe['fact'] == 0)])
""" x=z.drop(['fact','datetime_msc','tzoffset','severerisk','def_power'], axis = 1) """
x=z.drop(['fact','datetime_msc','tzoffset','severerisk'], axis = 1)
""" x = forecast_dataframe.drop(['fact','datetime_msc','tzoffset','severerisk','def_power'], axis = 1) """
""" print(x) """
y = z['fact']
""" y = forecast_dataframe['fact'] """
""" print(y) """
predict_dataframe=test_dataframe.drop(['datetime_msc','tzoffset','loadtime','sunrise','sunset','severerisk'], axis = 1)
""" predict_dataframe=test_dataframe.drop(['datetime_msc','tzoffset','loadtime','sunrise','sunset','severerisk','def_power'], axis = 1) """
""" print(predict_dataframe) """

x_train, x_validation, y_train, y_validation = train_test_split(x, y, train_size=0.8)

reg = catboost.CatBoostRegressor(depth = 10, boosting_type = 'Plain', bootstrap_type = 'Bayesian', learning_rate = 0.36195511036206707, bagging_temperature = 0.0824751595037756,
                        l2_leaf_reg = 0.07142313478321591, colsample_bylevel = 0.09800411259303943, min_data_in_leaf = 4, one_hot_max_size = 19,
                        loss_function = 'RMSE', cat_features = ['gtp','preciptype','conditions']).fit(x_train, y_train, eval_set=(x_validation, y_validation), silent = True)
predict = reg.predict(predict_dataframe)
test_dataframe['forecast'] = pd.DataFrame(predict)
feature_importance=reg.get_feature_importance(prettified=True)
print(feature_importance)

""" Обработка прогнозных значений """
""" Обрезаем по максимум за месяц в часы """
max_month_dataframe = pd.DataFrame()
for gtp in test_dataframe.gtp.value_counts().index:
    max_month=forecast_dataframe.loc[forecast_dataframe.gtp==gtp,['fact','hour','gtp']].groupby(by=['hour']).max()
    max_month_dataframe = max_month_dataframe.append(max_month, ignore_index=True)
max_month_dataframe['hour']=test_dataframe['hour']
""" max_month_dataframe.to_excel("max_month_dataframe.xlsx") """
test_dataframe=test_dataframe.merge(max_month_dataframe, left_on=['gtp','hour'], right_on = ['gtp','hour'], how='right')
test_dataframe.forecast[test_dataframe.forecast>test_dataframe.fact]=test_dataframe.fact[test_dataframe.forecast>test_dataframe.fact]
""" Обрезаем, если прогноз выше установленной мощности """
test_dataframe.forecast[test_dataframe.forecast>test_dataframe.def_power]=test_dataframe.def_power[test_dataframe.forecast>test_dataframe.def_power]
""" Если прогноз отрицательный, то 0 """
test_dataframe.forecast[test_dataframe.forecast<0]=0
test_dataframe.drop(['hour','fact'], axis='columns', inplace=True)
test_dataframe.to_excel("prediction_"+(datetime.datetime.today() + datetime.timedelta(days=1)).strftime("%d.%m.%Y")+".xlsx")

""" Запись прогноза в БД """
connection_vc = connection(0)
conn_cursor = connection_vc.cursor()
vall_predict = ''
for p in range(len(test_dataframe.index)):
    vall_predict= (vall_predict+"('"
    +str(test_dataframe.gtp[p])+"','"
    +str(test_dataframe.datetime_msc[p])+"','"
    +"18"+"','"                      
    +str(datetime.datetime.now().isoformat())+"','"                       
    +str(round(test_dataframe.forecast[p],3))+"'"+'),')  
vall_predict = vall_predict[:-1]
sql_predict = ("INSERT INTO visualcrossing.predict (gtp,dt,id_foreca,load_time,value) VALUES " + vall_predict + ";")
""" print(sql_predict) """
conn_cursor.execute(sql_predict) 
connection_vc.commit()
connection_vc.close()
""" Уведомление о подготовке прогноза """
telegram("VisualCrossing: прогноз подготовлен")



