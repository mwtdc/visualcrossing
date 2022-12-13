#!/usr/bin/python3.9
#!/usr/bin/env python
# coding: utf-8

import datetime
import json
import logging
import os
import pathlib
import urllib
import urllib.parse
import warnings
from sys import platform

import catboost
import numpy as np
import pandas as pd
import pymysql
import pyodbc
import requests
import yaml
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

start_time = datetime.datetime.now()
warnings.filterwarnings('ignore')

print('Visualcrossing Start!!!', datetime.datetime.now())

# Общий раздел
# Настройки для логера

logging.basicConfig(
    filename="log_journal_rsv.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s"
)

# Загружаем yaml файл с настройками

with open(
    f'{pathlib.Path(__file__).parent.absolute()}/settings.yaml', 'r'
) as yaml_file:
    settings = yaml.safe_load(yaml_file)
telegram_settings = pd.DataFrame(settings['telegram'])
sql_settings = pd.DataFrame(settings['sql_db'])
pyodbc_settings = pd.DataFrame(settings['pyodbc_db'])
vc_settings = pd.DataFrame(settings['visualcrossing_api'])

# Функция отправки уведомлений в telegram на любое количество каналов
# (указать данные в yaml файле настроек)


def telegram(text):
    msg = urllib.parse.quote(str(text))
    for channel in range(len(telegram_settings.index)):
        bot_token = str(telegram_settings.bot_token[channel])
        channel_id = str(telegram_settings.channel_id[channel])
        requests.get(f'https://api.telegram.org/bot'
                     f'{bot_token}/sendMessage?chat_id='
                     f'{channel_id}&text={msg}')

# Функция коннекта к базе Mysql
# (для выбора базы задать порядковый номер числом !начинается с 0!)


def connection(i):
    host_yaml = str(sql_settings.host[i])
    user_yaml = str(sql_settings.user[i])
    port_yaml = int(sql_settings.port[i])
    password_yaml = str(sql_settings.password[i])
    database_yaml = str(sql_settings.database[i])
    return pymysql.connect(
        host=host_yaml,
        user=user_yaml,
        port=port_yaml,
        password=password_yaml,
        database=database_yaml
    )

# Раздел загрузки прогноза погоды в базу

# Уведомление о старте задания
# telegram("VisualCrossing: загрузка прогноза запущена")

# Задаем переменные (даты для прогноза и токен для получения прогноза)


date_beg = (datetime.datetime.today()
            + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
date_end = (datetime.datetime.today()
            + datetime.timedelta(days=3)).strftime('%Y-%m-%d')
api_key = str(vc_settings.api_key[0])
x_test = pd.DataFrame()

# Загрузка списка ГТП и координат из базы

connection_geo = connection(0)
with connection_geo.cursor() as cursor:
    sql = 'select gtp,lat,lng from visualcrossing.ses_gtp;'
    cursor.execute(sql)
    ses_dataframe = pd.DataFrame(
        cursor.fetchall(),
        columns=['gtp', 'lat', 'lng']
    )
    connection_geo.close()

logging.info('Список ГТП и координаты станций загружены из базы.')

g = 0
for ses in range(len(ses_dataframe.index)):
    gtp = str(ses_dataframe.gtp[ses])
    lat = str(ses_dataframe.lat[ses]).replace(',', '.')
    lng = str(ses_dataframe.lng[ses]).replace(',', '.')
    print(gtp)
    try:
        url_response = requests.get(f'https://weather.visualcrossing.com/'
                                    f'VisualCrossingWebServices/rest/services/'
                                    f'timeline/{lat},{lng}/{date_beg}/'
                                    f'{date_end}?unitGroup=metric&key='
                                    f'{api_key}&include=fcst%2Chours&elements'
                                    f'=datetime,sunrise,sunset,temp,dew,'
                                    f'humidity,precip,precipprob,preciptype,'
                                    f'snow,snowdepth,windgust,windspeed,'
                                    f'pressure,cloudcover,visibility,'
                                    f'solarradiation,solarenergy,uvindex,'
                                    f'severerisk,conditions')
        url_response.raise_for_status()
        if url_response.ok:
            json_string = json.loads(url_response.text)
            dataframe_1 = pd.DataFrame(data=json_string['days'])
            g += 1
            print('Прогноз погоды загружен')
            logging.info(
                f'{g} Прогноз погоды для ГТП {gtp} загружен с visualcrossing'
                )
        else:
            print(
                f'VisualCrossing: Ошибка запроса: {url_response.status_code}'
            )
            logging.error(
                f'VisualCrossing: Ошибка запроса: {url_response.status_code}'
            )
            telegram(
                f'VisualCrossing: Ошибка запроса: {url_response.status_code}'
            )
            os._exit(1)
    except requests.HTTPError as http_err:
        print(
            f'VisualCrossing: HTTP error occurred: {http_err.response.text}'
        )
        logging.error(
            f'VisualCrossing: HTTP error occurred: {http_err.response.text}'
        )
        telegram(
            f'VisualCrossing: HTTP error occurred: {http_err.response.text}'
        )
        os._exit(1)
    except Exception as err:
        print(
            f'VisualCrossing: Other error occurred: {err}'
        )
        logging.error(
            f'VisualCrossing: Other error occurred: {err}'
        )
        telegram(
            f'VisualCrossing: Other error occurred: {err}'
        )
        os._exit(1)

    dataframe_3 = pd.DataFrame()
    for i in range(len(dataframe_1.index)):
        dataframe_2 = pd.DataFrame(data=dataframe_1['hours'][i])
        dataframe_2['datetime'] = (dataframe_1['datetime'][i]
                                   + ' ' + dataframe_2['datetime'])
        dataframe_2['sunrise'] = dataframe_1['sunrise'][i]
        dataframe_2['sunset'] = dataframe_1['sunset'][i]
        dataframe_2['gtp'] = gtp
        dataframe_2['tzoffset'] = json_string['tzoffset']
        tzoffset = dataframe_2.tzoffset[i]-3
        dataframe_2['datetime_msc'] = (pd.to_datetime(dataframe_2['datetime'], utc=False)
                                       - pd.DateOffset(hours=tzoffset))
        dataframe_3 = dataframe_3.append(dataframe_2, ignore_index=True)
    x_test = x_test.append(dataframe_3, ignore_index=True)

x_test.drop(
    x_test.index[np.where(x_test['datetime_msc'] < str(date_beg))[0]],
    inplace=True
)
x_test.reset_index(drop=True, inplace=True)

# Уведомление о количестве загруженных прогнозов
# telegram(f'VisualCrossing: загружен прогноз для {g} гтп')

logging.info(f'Сформирован датафрейм для {g} гтп')

connection_vc = connection(0)
conn_cursor = connection_vc.cursor()

vall = ''
rows = len(x_test.index)
gtp_rows = int(round(rows/72, 0))
for r in range(len(x_test.index)):
    vall = (vall+"('"
            + str(x_test.gtp[r])+"','"
            + str(x_test.datetime[r])+"','"
            + str(x_test.tzoffset[r])+"','"
            + str(x_test.datetime_msc[r])+"','"
            + str(datetime.datetime.now().isoformat())+"','"
            + str(x_test.sunrise[r])+"','"
            + str(x_test.sunset[r])+"','"
            + str(x_test.temp[r])+"','"
            + str(x_test.dew[r])+"','"
            + str(x_test.humidity[r])+"','"
            + str(x_test.precip[r]).replace('nan', '0.0')+"','"
            + str(x_test.precipprob[r]).replace('nan', '0.0')+"','"
            + str(x_test.preciptype[r]).replace(
                """['""", '').replace("""']""", '').replace(
                    """', '""", ",").replace(' ', 'None')+"','"
            + str(x_test.snow[r]).replace('nan', '0.0')+"','"
            + str(x_test.snowdepth[r])+"','"
            + str(x_test.windgust[r])+"','"
            + str(x_test.windspeed[r])+"','"
            + str(x_test.pressure[r])+"','"
            + str(x_test.cloudcover[r])+"','"
            + str(x_test.visibility[r])+"','"
            + str(x_test.solarradiation[r])+"','"
            + str(x_test.solarenergy[r]).replace(
                'nan', '0.0').replace(' ', '0.0').replace('NaN', '0.0')+"','"
            + str(x_test.uvindex[r])+"','"
            + str(x_test.severerisk[r])+"','"
            + str(x_test.conditions[r])+"'"+'),')

vall = vall[:-1]
sql = (f'INSERT INTO visualcrossing.forecast '
       f'(gtp,datetime,tzoffset,datetime_msc,loadtime,sunrise,sunset,temp,'
       f'dew,humidity,precip,precipprob,preciptype,snow,snowdepth,windgust,'
       f'windspeed,pressure,cloudcover,visibility,solarradiation,solarenergy,'
       f'uvindex,severerisk,conditions) VALUES {vall};')
conn_cursor.execute(sql)
connection_vc.commit()
connection_vc.close()

# Уведомление о записи в БД

telegram(f'VisualCrossing: записано в БД {rows} строк ({gtp_rows} гтп)')
logging.info(f'записано в БД {rows} строк прогноза погоды ({gtp_rows} гтп)')

# Раздел подготовки прогноза на catboost

# Загрузка прогноза погоды из базы и подготовка датафреймов

connection_geo = connection(0)
with connection_geo.cursor() as cursor:
    sql = 'select gtp,def_power from visualcrossing.ses_gtp;'
    cursor.execute(sql)
    ses_dataframe = pd.DataFrame(
        cursor.fetchall(),
        columns=['gtp', 'def_power']
    )
    ses_dataframe['def_power'] = ses_dataframe['def_power']*1000
    # ses_dataframe=ses_dataframe[ses_dataframe['gtp'].str.contains('GVIE', regex=False)]
    ses_dataframe = ses_dataframe[(ses_dataframe['gtp'].str.contains('GVIE', regex=False)) | 
                                  (ses_dataframe['gtp'].str.contains('GKZ', regex=False)) | 
                                  (ses_dataframe['gtp'].str.contains('GROZ', regex=False))]
    connection_geo.close()

connection_forecast = connection(0)
with connection_forecast.cursor() as cursor:
    sql = (f'select gtp,datetime_msc,tzoffset,loadtime,sunrise,sunset,temp,'
           f'dew,humidity,precip,precipprob,preciptype,snow,snowdepth,'
           f'windgust,windspeed,pressure,cloudcover,visibility,solarradiation,'
           f'solarenergy,uvindex,severerisk,conditions from '
           f'visualcrossing.forecast where loadtime >= CURDATE()'
           f' - INTERVAL 160 DAY;')
    cursor.execute(sql)
    forecast_dataframe = pd.DataFrame(
        cursor.fetchall(),
        columns=['gtp', 'datetime_msc', 'tzoffset', 'loadtime', 'sunrise',
                 'sunset', 'temp', 'dew', 'humidity', 'precip', 'precipprob',
                 'preciptype', 'snow', 'snowdepth', 'windgust', 'windspeed',
                 'pressure', 'cloudcover', 'visibility', 'solarradiation',
                 'solarenergy', 'uvindex', 'severerisk', 'conditions']
    )
    connection_forecast.close()

logging.info('Загружен массив прогноза погоды за предыдущие дни')

# Удаление дубликатов прогноза,
# т.к. каждый день грузит на 3 дня вперед и получается накладка

date_beg_predict = (datetime.datetime.today()
                    + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
date_end_predict = (datetime.datetime.today()
                    + datetime.timedelta(days=3)).strftime('%Y-%m-%d')
forecast_dataframe.drop_duplicates(
    subset=['datetime_msc', 'gtp'],
    keep='last',
    inplace=True,
    ignore_index=False
)
forecast_dataframe['hour'] = pd.to_datetime(forecast_dataframe.datetime_msc.values).hour
forecast_dataframe['month'] = pd.to_datetime(forecast_dataframe.datetime_msc.values).month
forecast_dataframe['cloudcover'] = (100
                                    - forecast_dataframe['cloudcover'].astype('float'))
test_dataframe = forecast_dataframe.drop(
    forecast_dataframe.index[
        np.where(forecast_dataframe['datetime_msc'] < str(date_beg_predict))[0]
        ]
)
test_dataframe.drop(
    forecast_dataframe.index[
        np.where(forecast_dataframe['datetime_msc'] > str(date_end_predict))[0]
        ],
    inplace=True
)
# test_dataframe=test_dataframe[test_dataframe['gtp'].str.contains('GVIE', regex=False)]
test_dataframe = test_dataframe.merge(
    ses_dataframe,
    left_on=['gtp'],
    right_on=['gtp'],
    how='left'
)
forecast_dataframe.drop(
    forecast_dataframe.index[
        np.where(
            forecast_dataframe['datetime_msc'] > str(datetime.datetime.today())
            )[0]
        ],
    inplace=True
)

# Сортировка датафрейма по гтп и дате

forecast_dataframe.sort_values(['gtp', 'datetime_msc'], inplace=True)
forecast_dataframe['datetime_msc'] = forecast_dataframe['datetime_msc'].astype('datetime64[ns]')

logging.info('forecast_dataframe и test_dataframe преобразованы в нужный вид')

# Загрузка факта выработки
server = str(pyodbc_settings.host[0])
database = str(pyodbc_settings.database[0])
username = str(pyodbc_settings.user[0])
password = str(pyodbc_settings.password[0])
if platform == "linux" or platform == "linux2":
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
elif platform == "win32":
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

cursor.execute("SELECT SUBSTRING (... ,"
               "len(...)-8, 8) as gtp, MIN(DT) as DT,"
               " SUM(Val) as Val FROM ... JOIN ... ON"
               " ...=... JOIN ... ON"
               " ...=... WHERE ... like"
               " 'Генерация%{...%' AND ...=... AND DT >= DATEADD(HOUR, -260"
               " * 24, DATEDIFF(d, 0, GETDATE()))"
               " AND ... NOT LIKE ..."
               " GROUP BY SUBSTRING (... ,"
               "len(...)-8, 8), DATEPART(YEAR, DT),"
               " DATEPART(MONTH, DT), DATEPART(DAY, DT), DATEPART(HOUR, DT)"
               " ORDER BY SUBSTRING (... ,"
               "len(...)-8, 8), DATEPART(YEAR, DT),"
               " DATEPART(MONTH, DT), DATEPART(DAY, DT), DATEPART(HOUR, DT);")
fact = cursor.fetchall()
cnxn.close()

logging.info('Загружен факт выработки из БД')

fact = pd.DataFrame(np.array(fact), columns=['gtp', 'dt', 'fact'])
fact.drop_duplicates(
    subset=['gtp', 'dt'],
    keep='last',
    inplace=True,
    ignore_index=False
)
forecast_dataframe = forecast_dataframe.merge(
    ses_dataframe,
    left_on=['gtp'],
    right_on=['gtp'],
    how='left'
)
forecast_dataframe = forecast_dataframe.merge(
    fact,
    left_on=['gtp', 'datetime_msc'],
    right_on=['gtp', 'dt'],
    how='left'
)
forecast_dataframe.dropna(subset=['fact'], inplace=True)
forecast_dataframe.drop(
    ['dt', 'loadtime', 'sunrise', 'sunset'],
    axis='columns',
    inplace=True
)
# forecast_dataframe.to_excel('forecast_dataframe2.xlsx')
# test_dataframe.to_excel('test_dataframe2.xlsx')

logging.info('Датафреймы погоды и факта выработки склеены')

# CatBoost


z = forecast_dataframe.drop(
    forecast_dataframe.index[np.where(forecast_dataframe['fact'] == 0)]
)
z['gtp'] = z['gtp'].str.replace('GVIE', '1')
z['gtp'] = z['gtp'].str.replace('GKZV', '4')
z['gtp'] = z['gtp'].str.replace('GKZ', '2')
z['gtp'] = z['gtp'].str.replace('GROZ', '3')
x = z.drop(
    ['preciptype', 'conditions', 'fact',
     'datetime_msc', 'tzoffset', 'severerisk'], axis=1
    )
x['gtp'] = x['gtp'].astype('int')
y = z['fact']

predict_dataframe = test_dataframe.drop(
    ['preciptype', 'conditions', 'datetime_msc', 'tzoffset',
     'loadtime', 'sunrise', 'sunset', 'severerisk'], axis=1
     )
predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GVIE', '1')
predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GKZV', '4')
predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GKZ', '2')
predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GROZ', '3')
predict_dataframe['gtp'] = predict_dataframe['gtp'].astype('int')

x_train, x_validation, y_train, y_validation = train_test_split(x, y, train_size=0.8)

logging.info('Старт предикта на CatBoostRegressor')

reg = catboost.CatBoostRegressor(
    depth=10, boosting_type='Plain', bootstrap_type='MVS',
    learning_rate=0.5167293015679837, l2_leaf_reg=0.03823029591700387,
    colsample_bylevel=0.09974262823193879, min_data_in_leaf=17,
    one_hot_max_size=16, loss_function='RMSE', verbose=0
    )
regr = BaggingRegressor(
    base_estimator=reg,
    n_estimators=50,
    n_jobs=-1,
    random_state=118).fit(x_train, y_train)
predict = regr.predict(predict_dataframe)
test_dataframe['forecast'] = pd.DataFrame(predict)
# feature_importance=reg.get_feature_importance(prettified=True)
# print(feature_importance)

logging.info('Подготовлен прогноз на CatBoostRegressor')

# Обработка прогнозных значений
# Обрезаем по максимум за месяц в часы


max_month_dataframe = pd.DataFrame()
date_cut = (datetime.datetime.today()
            + datetime.timedelta(days=-29)).strftime('%Y-%m-%d')
cut_dataframe = forecast_dataframe.drop(
    forecast_dataframe.index[
        np.where(forecast_dataframe['datetime_msc'] < str(date_cut))[0]
        ]
)
for gtp in test_dataframe.gtp.value_counts().index:
    max_month = cut_dataframe.loc[
        cut_dataframe.gtp==gtp,
        ['fact', 'hour', 'gtp']].groupby(by=['hour']).max()
    max_month_dataframe = max_month_dataframe.append(
        max_month,
        ignore_index=True
    )
max_month_dataframe['hour'] = test_dataframe['hour']
test_dataframe = test_dataframe.merge(
    max_month_dataframe,
    left_on=['gtp', 'hour'],
    right_on=['gtp', 'hour'],
    how='left'
)
test_dataframe['forecast'] = test_dataframe[
    ['forecast', 'fact', 'def_power']].min(axis=1)

# Если прогноз отрицательный, то 0

test_dataframe.forecast[test_dataframe.forecast < 0] = 0
test_dataframe.drop(
    ['hour', 'month', 'fact'],
    axis='columns',
    inplace=True
)
test_dataframe.to_excel(f'/var/log/visualcrossing/arhive/prediction_'
                        f'{(datetime.datetime.today() + datetime.timedelta(days=1)).strftime("%d.%m.%Y")}.xlsx')

logging.info('Датафрейм с прогнозом выработки прошел обработку от'
             ' нулевых значений и обрезку по макс за месяц')

# Запись прогноза в БД

connection_vc = connection(2)
conn_cursor = connection_vc.cursor()
vall_predict = ''
for p in range(len(test_dataframe.index)):
    vall_predict = (vall_predict+"('"
                    + str(test_dataframe.gtp[p])+"','"
                    + str(test_dataframe.datetime_msc[p])+"','"
                    + "18"+"','"
                    + str(datetime.datetime.now().isoformat())+"','"
                    + str(round(test_dataframe.forecast[p], 3))+"'"+'),')
vall_predict = vall_predict[:-1]
sql_predict = (f'INSERT INTO weather_foreca '
               f'(gtp,dt,id_foreca,load_time,value) VALUES {vall_predict};')
conn_cursor.execute(sql_predict)
connection_vc.commit()
connection_vc.close()

# Уведомление о подготовке прогноза

telegram('VisualCrossing: прогноз подготовлен')
logging.info('Прогноз записан в БД treid_03')
print('Время выполнения:', datetime.datetime.now() - start_time)
