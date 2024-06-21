#!/usr/bin/python3.9
#!/usr/bin/env python
# coding: utf-8

import datetime
import json
import logging
import pathlib
import urllib
import urllib.parse
import warnings
from sys import platform
from time import sleep

import catboost
import numpy as np
import optuna
import pandas as pd
import pyodbc
import requests
import yaml
from catboost import CatBoostRegressor
from optuna.samplers import TPESampler
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

# Коэффициент завышения прогноза:
OVERVALUE_COEFF = 1.06
# Задаем переменные (даты для прогноза и список погодных параметров)
DATE_BEG = (datetime.datetime.today() + datetime.timedelta(days=1)).strftime(
    "%Y-%m-%d"
)
DATE_END = (datetime.datetime.today() + datetime.timedelta(days=3)).strftime(
    "%Y-%m-%d"
)
DATE_BEG_PREDICT = (
    datetime.datetime.today() + datetime.timedelta(days=1)
).strftime("%Y-%m-%d")
DATE_END_PREDICT = (
    datetime.datetime.today() + datetime.timedelta(days=3)
).strftime("%Y-%m-%d")

COL_PARAMETERS = [
    "datetime",
    "sunrise",
    "sunset",
    "temp",
    "dew",
    "humidity",
    "precip",
    "precipprob",
    "preciptype",
    "snow",
    "snowdepth",
    "windgust",
    "windspeed",
    "pressure",
    "cloudcover",
    "visibility",
    "solarradiation",
    "solarenergy",
    "uvindex",
    "severerisk",
    "conditions",
]

DB_COLUMNS = [
    "gtp",
    "datetime",
    "tzoffset",
    "datetime_msc",
    "loadtime",
    "sunrise",
    "sunset",
    "temp",
    "dew",
    "humidity",
    "precip",
    "precipprob",
    "preciptype",
    "snow",
    "snowdepth",
    "windgust",
    "windspeed",
    "pressure",
    "cloudcover",
    "visibility",
    "solarradiation",
    "solarenergy",
    "uvindex",
    "severerisk",
    "conditions",
]


start_time = datetime.datetime.now()
warnings.filterwarnings("ignore")


print("Visualcrossing Start!!!", datetime.datetime.now())


# Общий раздел
# Настройки для логера
if platform == "linux" or platform == "linux2":
    logging.basicConfig(
        filename="/var/log/log-execute/visualcrossing_log_journal_rsv.log.txt",
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(funcName)s: %(lineno)d - %(message)s"
        ),
    )
elif platform == "win32":
    logging.basicConfig(
        filename=(
            f"{pathlib.Path(__file__).parent.absolute()}"
            "/visualcrossing_log_journal_rsv.log.txt"
        ),
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(funcName)s: %(lineno)d - %(message)s"
        ),
    )

# Загружаем yaml файл с настройками
with open(
    str(pathlib.Path(__file__).parent.absolute()) + "/settings.yaml", "r"
) as yaml_file:
    settings = yaml.safe_load(yaml_file)
telegram_settings = pd.DataFrame(settings["telegram"])
sql_settings = pd.DataFrame(settings["sql_db"])
pyodbc_settings = pd.DataFrame(settings["pyodbc_db"])
vc_settings = pd.DataFrame(settings["visualcrossing_api"])

API_KEY = str(vc_settings.api_key[0])


# Функция отправки уведомлений в telegram на любое количество каналов
# (указать данные в yaml файле настроек)
def telegram(i, text):
    try:
        msg = urllib.parse.quote(str(text))
        bot_token = str(telegram_settings.bot_token[i])
        channel_id = str(telegram_settings.channel_id[i])

        retry_strategy = Retry(
            total=3,
            status_forcelist=[101, 429, 500, 502, 503, 504],
            method_whitelist=["GET", "POST"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        http.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={channel_id}&text={msg}",
            verify=False,
            timeout=10,
        )
    except Exception as err:
        print(f"VisualCrossing: Ошибка при отправке в telegram -  {err}")
        logging.error(
            f"VisualCrossing: Ошибка при отправке в telegram -  {err}"
        )


# Функция коннекта к базе Mysql
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)
def connection(i):
    host_yaml = str(sql_settings.host[i])
    user_yaml = str(sql_settings.user[i])
    port_yaml = int(sql_settings.port[i])
    password_yaml = str(sql_settings.password[i])
    database_yaml = str(sql_settings.database[i])
    db_data = f"mysql://{user_yaml}:{password_yaml}@{host_yaml}:{port_yaml}/{database_yaml}"
    return create_engine(db_data).connect()


# Функция загрузки факта выработки
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)
def fact_load(i, dt):
    server = str(pyodbc_settings.host[i])
    database = str(pyodbc_settings.database[i])
    username = str(pyodbc_settings.user[i])
    password = str(pyodbc_settings.password[i])
    # Выбор драйвера в зависимости от ОС
    if platform == "linux" or platform == "linux2":
        connection_ms = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
    elif platform == "win32":
        connection_ms = pyodbc.connect(
            "DRIVER={SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
    #
    mssql_cursor = connection_ms.cursor()
    mssql_cursor.execute(
        "SELECT SUBSTRING (Points.PointName ,"
        "len(Points.PointName)-8, 8) as gtp, MIN(DT) as DT,"
        " SUM(Val) as Val FROM Points JOIN PointParams ON "
        "Points.ID_Point=PointParams.ID_Point JOIN PointMains"
        " ON PointParams.ID_PP=PointMains.ID_PP WHERE "
        "PointName like 'Генерация%{G%' AND ID_Param=153 "
        "AND DT >= "
        + str(dt)
        + " AND PointName NOT LIKE "
        "'%GVIE0001%' AND PointName NOT LIKE '%GVIE0012%' "
        "AND PointName NOT LIKE '%GVIE0416%' AND PointName "
        "NOT LIKE '%GVIE0167%' "
        "AND PointName NOT LIKE '%GVIE0007%' "
        "AND PointName "
        "NOT LIKE '%GVIE0987%' AND PointName NOT LIKE "
        "'%GVIE0988%' AND PointName NOT LIKE '%GVIE0989%' "
        "AND PointName NOT LIKE '%GVIE0991%' AND PointName "
        "NOT LIKE '%GVIE0994%' AND PointName NOT LIKE "
        "'%GVIE1372%' GROUP BY SUBSTRING (Points.PointName "
        ",len(Points.PointName)-8, 8), DATEPART(YEAR, DT), "
        "DATEPART(MONTH, DT), DATEPART(DAY, DT), "
        "DATEPART(HOUR, DT) ORDER BY SUBSTRING "
        "(Points.PointName ,len(Points.PointName)-8, 8), "
        "DATEPART(YEAR, DT), DATEPART(MONTH, DT), "
        "DATEPART(DAY, DT), DATEPART(HOUR, DT);"
    )
    fact = mssql_cursor.fetchall()
    connection_ms.close()
    fact = pd.DataFrame(np.array(fact), columns=["gtp", "dt", "fact"])
    fact.drop_duplicates(
        subset=["gtp", "dt"], keep="last", inplace=True, ignore_index=False
    )
    fact["fact"] = fact["fact"].astype("float").round(-2)
    return fact


# Функция записи датафрейма в базу
def load_data_to_db(db_name, connect_id, dataframe):
    telegram(1, "Visualcrossing: Старт записи в БД.")
    logging.info("Visualcrossing: Старт записи в БД.")

    dataframe = pd.DataFrame(dataframe)
    connection_skm = connection(connect_id)
    dataframe.to_sql(
        name=db_name,
        con=connection_skm,
        if_exists="append",
        index=False,
        chunksize=5000,
    )
    rows = len(dataframe)
    telegram(
        1, f"Visualcrossing: записано в БД {rows} строк ({int(rows/72)} гтп)"
    )
    if len(dataframe.columns) > 5:
        telegram(
            0,
            f"Visualcrossing: записано в БД {rows} строк ({int(rows/72)} гтп)",
        )
    logging.info(f"записано в БД {rows} строк c погодой ({int(rows/72)} гтп)")
    telegram(1, "Visualcrossing: Финиш записи в БД.")
    logging.info("Visualcrossing: Финиш записи в БД.")


# Функция загрузки датафрейма из базы
def load_data_from_db(
    db_name,
    col_from_database,
    connect_id,
    condition_column,
    day_interval,
):
    telegram(1, "Visualcrossing: Старт загрузки из БД.")
    logging.info("Visualcrossing: Старт загрузки из БД.")

    list_col_database = ",".join(col_from_database)
    connection_db = connection(connect_id)
    if day_interval is None and condition_column is None:
        query = f"select {list_col_database} from {db_name};"
    else:
        query = (
            f"select {list_col_database} from {db_name} where"
            f" {condition_column} >= CURDATE() - INTERVAL {day_interval} DAY;"
        )
    dataframe_from_db = pd.read_sql(sql=query, con=connection_db)

    telegram(1, "Visualcrossing: Финиш загрузки из БД.")
    logging.info("Visualcrossing: Финиш загрузки из БД.")
    return dataframe_from_db


# Раздел загрузки прогноза погоды в базу
def load_forecast_to_db(date_beg, date_end, api_key, col_parameters):
    telegram(1, "Visualcrossing: Старт загрузки погоды.")

    list_parameters = ",".join(col_parameters)
    weather_dataframe = pd.DataFrame()

    # Загрузка списка ГТП и координат из базы

    ses_dataframe = load_data_from_db(
        "visualcrossing.ses_gtp",
        ["gtp", "lat", "lng"],
        0,
        None,
        None,
    )

    # Ниже можно выбирать гтп в датафрейме, только опт, кз, розн или все.
    # ses_dataframe = ses_dataframe[
    #     ses_dataframe["gtp"].str.contains("GK", regex=False)
    # ]
    ses_dataframe = ses_dataframe[
        (ses_dataframe["gtp"].str.contains("GVIE", regex=False))
        | (ses_dataframe["gtp"].str.contains("GKZ0", regex=False))
        | (ses_dataframe["gtp"].str.contains("GROZ", regex=False))
    ]
    ses_dataframe.reset_index(inplace=True)

    logging.info(
        f"Список ГТП и координаты станций загружены из базы"
        f" visualcrossing.ses_gtp"
    )

    g = 0
    for ses in range(len(ses_dataframe.index)):
        gtp = str(ses_dataframe.gtp[ses])
        lat = str(ses_dataframe.lat[ses]).replace(",", ".")
        lng = str(ses_dataframe.lng[ses]).replace(",", ".")
        print(gtp)
        try:
            url_response = requests.get(
                f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lng}/{date_beg}/{date_end}?unitGroup=metric&key={api_key}&include=fcst%2Chours&elements={list_parameters}",
                verify=False,
            )
            url_response.raise_for_status()
            while url_response.status_code != 200:
                url_response = requests.get(
                    f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lng}/{date_beg}/{date_end}?unitGroup=metric&key={api_key}&include=fcst%2Chours&elements={list_parameters}",
                    verify=False,
                )
                sleep(20)
            if url_response.ok:
                json_string = json.loads(url_response.text)
                dataframe_1 = pd.DataFrame(data=json_string["days"])
                g += 1
                print("прогноз погоды загружен")
                logging.info(
                    f"{g} Прогноз погоды для ГТП {gtp} загружен с"
                    " visualcrossing.com"
                )
            else:
                print(
                    "VisualCrossing: Ошибка запроса:"
                    f" {url_response.status_code}"
                )
                logging.error(
                    "VisualCrossing: Ошибка запроса:"
                    f" {url_response.status_code}"
                )
                telegram(
                    1,
                    "VisualCrossing: Ошибка запроса:"
                    f" {url_response.status_code}",
                )
                # os._exit(1)
        except requests.HTTPError as http_err:
            print(
                "VisualCrossing: HTTP error occurred:"
                f" {http_err.response.text}"
            )
            logging.error(
                "VisualCrossing: HTTP error occurred:"
                f" {http_err.response.text}"
            )
            telegram(
                1,
                "VisualCrossing: HTTP error occurred:"
                f" {http_err.response.text}",
            )
            # os._exit(1)
        except Exception as err:
            print(f"VisualCrossing: Other error occurred: {err}")
            logging.error(f"VisualCrossing: Other error occurred: {err}")
            telegram(1, f"VisualCrossing: Other error occurred: {err}")
            # os._exit(1)

        dataframe_3 = pd.DataFrame()
        for i in range(len(dataframe_1.index)):
            dataframe_2 = pd.DataFrame(data=dataframe_1["hours"][i])
            dataframe_2["datetime"] = (
                dataframe_1["datetime"][i] + " " + dataframe_2["datetime"]
            )
            dataframe_2["sunrise"] = dataframe_1["sunrise"][i]
            dataframe_2["sunset"] = dataframe_1["sunset"][i]
            dataframe_2["gtp"] = gtp
            dataframe_2["tzoffset"] = json_string["tzoffset"]
            tzoffset = dataframe_2.tzoffset[i] - 3
            dataframe_2["datetime_msc"] = pd.to_datetime(
                dataframe_2["datetime"], utc=False
            ) - pd.DateOffset(hours=tzoffset)
            dataframe_2["loadtime"] = datetime.datetime.now().isoformat()

            for i in range(len(DB_COLUMNS)):
                dataframe_2.insert(
                    i, DB_COLUMNS[i], dataframe_2.pop(DB_COLUMNS[i])
                )

            dataframe_3 = dataframe_3.append(dataframe_2, ignore_index=True)
            dataframe_3["preciptype"] = (
                dataframe_3["preciptype"]
                .astype("str")
                .str.replace("['", "", regex=False)
            )
            dataframe_3["preciptype"] = (
                dataframe_3["preciptype"]
                .astype("str")
                .str.replace("']", "", regex=False)
            )
            dataframe_3["preciptype"] = (
                dataframe_3["preciptype"]
                .astype("str")
                .str.replace("', '", ",", regex=False)
            )
            dataframe_3["preciptype"] = (
                dataframe_3["preciptype"]
                .astype("str")
                .str.replace(" ", "None", regex=False)
            )
            dataframe_3["solarenergy"] = (
                dataframe_3["solarenergy"]
                .astype("str")
                .str.replace(" ", "0.0", regex=False)
            )
            dataframe_3.fillna(0, inplace=True)
            # print(dataframe_3)
        weather_dataframe = weather_dataframe.append(
            dataframe_3, ignore_index=True
        )
    # print(weather_dataframe)
    weather_dataframe.drop(
        weather_dataframe.index[
            np.where(weather_dataframe["datetime_msc"] < str(DATE_BEG))[0]
        ],
        inplace=True,
    )
    weather_dataframe.reset_index(drop=True, inplace=True)

    logging.info("Сформирован датафрейм для " + str(g) + " гтп")

    load_data_to_db(
        "forecast",
        0,
        weather_dataframe,
    )


# Загрузка прогнозов погоды по станциям из базы и подготовка датафреймов
def prepare_datasets_to_train():
    ses_dataframe = load_data_from_db(
        "visualcrossing.ses_gtp",
        ["gtp", "def_power"],
        0,
        None,
        None,
    )
    ses_dataframe["def_power"] = ses_dataframe["def_power"] * 1000
    # ses_dataframe = ses_dataframe[
    #     ses_dataframe["gtp"].str.contains("GVIE", regex=False)
    # ]
    ses_dataframe = ses_dataframe[
        (ses_dataframe["gtp"].str.contains("GVIE", regex=False))
        | (ses_dataframe["gtp"].str.contains("GKZ0", regex=False))
        | (ses_dataframe["gtp"].str.contains("GROZ", regex=False))
    ]
    logging.info("Загружен датафрейм с гтп и установленной мощностью.")

    forecast_dataframe = load_data_from_db(
        "visualcrossing.forecast",
        DB_COLUMNS,
        0,
        "loadtime",
        365,
    )
    forecast_dataframe.drop(
        [
            "datetime",
        ],
        axis="columns",
        inplace=True,
    )
    
    logging.info("Загружен массив прогноза погоды за предыдущие дни")

    # Удаление дубликатов прогноза,
    # т.к. каждый день грузит на 3 дня вперед и получается накладка
    forecast_dataframe.drop_duplicates(
        subset=["datetime_msc", "gtp"],
        keep="last",
        inplace=True,
        ignore_index=False,
    )
    forecast_dataframe["hour"] = pd.to_datetime(
        forecast_dataframe.datetime_msc.values
    ).hour
    forecast_dataframe["month"] = pd.to_datetime(
        forecast_dataframe.datetime_msc.values
    ).month
    # forecast_dataframe['cloudcover']=100-forecast_dataframe['cloudcover'].astype('float')
    test_dataframe = forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(
                forecast_dataframe["datetime_msc"] < str(DATE_BEG_PREDICT)
            )[0]
        ]
    )
    test_dataframe.drop(
        forecast_dataframe.index[
            np.where(
                forecast_dataframe["datetime_msc"] > str(DATE_END_PREDICT)
            )[0]
        ],
        inplace=True,
    )
    # test_dataframe=test_dataframe[test_dataframe['gtp'].str.contains('GVIE', regex=False)]
    test_dataframe = test_dataframe.merge(
        ses_dataframe, left_on=["gtp"], right_on=["gtp"], how="left"
    )

    forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(
                forecast_dataframe["datetime_msc"]
                > str(datetime.datetime.today())
            )[0]
        ],
        inplace=True,
    )
    # Сортировка датафрейма по гтп и дате
    forecast_dataframe.sort_values(["gtp", "datetime_msc"], inplace=True)
    forecast_dataframe["datetime_msc"] = forecast_dataframe[
        "datetime_msc"
    ].astype("datetime64[ns]")
    logging.info(
        "forecast_dataframe и test_dataframe преобразованы в нужный вид"
    )

    fact = fact_load(0, "DATEADD(HOUR, -365 * 24, DATEDIFF(d, 0, GETDATE()))")

    forecast_dataframe = forecast_dataframe.merge(
        ses_dataframe, left_on=["gtp"], right_on=["gtp"], how="left"
    )
    forecast_dataframe = forecast_dataframe.merge(
        fact,
        left_on=["gtp", "datetime_msc"],
        right_on=["gtp", "dt"],
        how="left",
    )
    forecast_dataframe.dropna(subset=["fact"], inplace=True)
    forecast_dataframe.drop(
        ["dt", "loadtime", "sunrise", "sunset"], axis="columns", inplace=True
    )
    logging.info("Датафреймы погоды и факта выработки склеены")
    return forecast_dataframe, test_dataframe


# Раздел подготовки прогноза на CatBoost
def prepare_forecast_catboost(forecast_dataframe, test_dataframe):
    z = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(forecast_dataframe["fact"] == 0)]
    )
    z["gtp"] = z["gtp"].str.replace("GVIE", "1")
    z["gtp"] = z["gtp"].str.replace("GKZV", "4")
    z["gtp"] = z["gtp"].str.replace("GKZ", "2")
    z["gtp"] = z["gtp"].str.replace("GROZ", "3")
    x = z.drop(
        [
            "preciptype",
            "conditions",
            "fact",
            "datetime_msc",
            "tzoffset",
            "severerisk",
        ],
        axis=1,
    )
    x["gtp"] = x["gtp"].astype("int")
    y = z["fact"]

    predict_dataframe = test_dataframe.drop(
        [
            "preciptype",
            "conditions",
            "datetime_msc",
            "tzoffset",
            "loadtime",
            "sunrise",
            "sunset",
            "severerisk",
        ],
        axis=1,
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GVIE", "1"
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GKZV", "4"
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace("GKZ", "2")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GROZ", "3"
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].astype("int")

    logging.info("Старт предикта на CatBoostRegressor")
    reg = catboost.CatBoostRegressor(
        depth=10,
        boosting_type="Plain",
        bootstrap_type="MVS",
        learning_rate=0.5167293015679837,
        l2_leaf_reg=0.03823029591700387,
        colsample_bylevel=0.09974262823193879,
        min_data_in_leaf=17,
        one_hot_max_size=16,
        loss_function="RMSE",
        verbose=0,
    )
    regr = BaggingRegressor(
        base_estimator=reg, n_estimators=50, n_jobs=-1, random_state=118
    ).fit(x, y)
    predict = regr.predict(predict_dataframe)
    test_dataframe["forecast"] = pd.DataFrame(predict)
    test_dataframe["forecast"] = test_dataframe["forecast"] * OVERVALUE_COEFF

    # feature_importance=reg.get_feature_importance(prettified=True)
    # print(feature_importance)
    logging.info("Подготовлен прогноз на CatBoostRegressor")

    # Обработка прогнозных значений
    # Обрезаем по максимум за месяц в часы
    max_month_dataframe = pd.DataFrame()
    date_cut = (
        datetime.datetime.today() + datetime.timedelta(days=-29)
    ).strftime("%Y-%m-%d")
    cut_dataframe = forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(forecast_dataframe["datetime_msc"] < str(date_cut))[0]
        ]
    )
    for gtp in test_dataframe.gtp.value_counts().index:
        max_month = (
            cut_dataframe.loc[
                cut_dataframe.gtp == gtp, ["fact", "hour", "gtp"]
            ]
            .groupby(by=["hour"])
            .max()
        )
        max_month.reset_index(inplace=True)
        max_month_dataframe = max_month_dataframe.append(
            max_month, ignore_index=True
        )
    # max_month_dataframe['hour']=test_dataframe['hour']
    test_dataframe = test_dataframe.merge(
        max_month_dataframe,
        left_on=["gtp", "hour"],
        right_on=["gtp", "hour"],
        how="left",
    )
    test_dataframe.fillna(0, inplace=True)
    test_dataframe["forecast"] = test_dataframe[
        ["forecast", "fact", "def_power"]
    ].min(axis=1)
    # Если прогноз отрицательный, то 0
    test_dataframe.forecast[test_dataframe.forecast < 0] = 0
    test_dataframe["forecast"] = np.where(
        test_dataframe["forecast"] == 0,
        (
            np.where(
                test_dataframe["fact"] > 0, np.NaN, test_dataframe.forecast
            )
        ),
        test_dataframe.forecast,
    )
    test_dataframe["forecast"].interpolate(
        method="linear", axis=0, inplace=True
    )
    test_dataframe["forecast"] = test_dataframe[["forecast", "fact"]].min(
        axis=1
    )
    test_dataframe.drop(
        ["hour", "month", "fact"], axis="columns", inplace=True
    )

    test_dataframe.drop(
        COL_PARAMETERS
        + [
            "loadtime",
            "def_power",
            "tzoffset",
        ],
        axis="columns",
        inplace=True,
        errors="ignore",
    )

    # Добавить к датафрейму столбцы с текущей датой и id прогноза
    # INSERT INTO treid_03.weather_foreca (gtp,dt,id_foreca,load_time,value)
    test_dataframe.insert(2, "id_foreca", "18")
    test_dataframe.insert(3, "load_time", datetime.datetime.now().isoformat())
    test_dataframe.rename(
        columns={"datetime_msc": "dt", "forecast": "value"},
        errors="raise",
        inplace=True,
    )

    logging.info(
        f"Датафрейм с прогнозом выработки прошел обработку"
        f" от нулевых значений и обрезку по макс за месяц"
    )

    # Запись прогноза в БД
    load_data_to_db("weather_foreca", 1, test_dataframe)

    # Уведомление о подготовке прогноза
    telegram(0, "VisualCrossing: прогноз подготовлен")
    logging.info("Прогноз записан в БД treid_03")


def optuna_tune_params(forecast_dataframe, test_dataframe):
    # Подбор параметров через Optuna
    z = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(forecast_dataframe["fact"] == 0)]
    )
    z["gtp"] = z["gtp"].str.replace("GVIE", "1")
    z["gtp"] = z["gtp"].str.replace("GKZV", "4")
    z["gtp"] = z["gtp"].str.replace("GKZ", "2")
    z["gtp"] = z["gtp"].str.replace("GROZ", "3")
    x = z.drop(
        [
            "preciptype",
            "conditions",
            "fact",
            "datetime_msc",
            "tzoffset",
            "severerisk",
        ],
        axis=1,
    )
    x["gtp"] = x["gtp"].astype("int")
    y = z["fact"]

    predict_dataframe = test_dataframe.drop(
        [
            "preciptype",
            "conditions",
            "datetime_msc",
            "tzoffset",
            "loadtime",
            "sunrise",
            "sunset",
            "severerisk",
        ],
        axis=1,
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GVIE", "1"
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GKZV", "4"
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace("GKZ", "2")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GROZ", "3"
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].astype("int")

    def objective(trial):
        x_train, x_validation, y_train, y_validation = train_test_split(
            x, y, train_size=0.8
        )
        # 'tree_method':'gpu_hist',
        # this parameter means using the GPU when training our model
        # to speedup the training process
        param = {
            "loss_function": trial.suggest_categorical(
                "loss_function", ["RMSE", "MAE"]
            ),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 1e-5, 1e0
            ),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 1e0),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.01, 0.1
            ),
            "depth": trial.suggest_int("depth", 1, 10),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
        }
        # Conditional Hyper-Parameters
        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10
            )
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        reg = CatBoostRegressor(**param)
        reg.fit(
            x_train,
            y_train,
            eval_set=(x_validation, y_validation),
            verbose=0,
            early_stopping_rounds=200,
        )
        prediction = reg.predict(predict_dataframe)
        score = reg.score(x_train, y_train)
        return score

    study = optuna.create_study(sampler=TPESampler(), direction="maximize")
    study.optimize(objective, n_trials=1000, timeout=3600)
    optuna_vis = optuna.visualization.plot_param_importances(study)
    print(optuna_vis)
    print("Number of completed trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("\tBest Score: {}".format(trial.value))
    print("\tBest Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# Сам процесс работы разбит для удобства по функциям
# чтобы если погода загрузилась, а прогноз не подготовился,
#  то чтобы не тратить лимит запросов и не засорять базу,
# закомменчиваем первую функцию и разбираемся дальше сколько угодно попыток.
# 1 - load_forecast_to_db - загрузка прогноза с сайта и запись в бд
# 2 - prepare_datasets_to_train - подготовка датасетов для обучения модели,
# переменным присваиваются возвращаемые 2 датафрейма и список столбцов,
# необходимо для работы следующих функций.
# 3 - optuna_tune_params - подбор параметров для модели через оптуну
# необходимо в нее передать 2 датафрейма из предыдущей функции.
# 4 - prepare_forecast_xgboost - подготовка прогноза,


# # 1
load_forecast_to_db(DATE_BEG, DATE_END, API_KEY, COL_PARAMETERS)
# # 2
forecast_dataframe, test_dataframe = prepare_datasets_to_train()
# # 3
# optuna_tune_params(forecast_dataframe, test_dataframe)
# # 4
prepare_forecast_catboost(forecast_dataframe, test_dataframe)

print("Время выполнения:", datetime.datetime.now() - start_time)
