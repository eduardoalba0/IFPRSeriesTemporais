import numpy as np
import pandas as pd


def prepararDados():
    df_agua = pd.read_csv("./Dados/ConsumoAguaIFPR.csv", delimiter=";")
    df_energia = pd.read_csv("./Dados/ConsumoEnergiaIFPR.csv", delimiter=";")
    df_clima = pd.read_csv("./Dados/ClimaAgritempo.csv", delimiter=";")
    df_horas_aula = pd.read_csv("./Dados/HorasAulaIFPR.csv", delimiter=";")
    df_agua["DATA"] = pd.to_datetime(df_agua["DATA"], format="%d/%m/%Y")
    df_energia["DATA"] = pd.to_datetime(df_energia["DATA"], format="%d/%m/%Y")
    df_clima["DATA"] = pd.to_datetime((df_clima["DATA"]), format="%d/%m/%Y")
    df_horas_aula["DATA"] = pd.to_datetime((df_horas_aula["DATA"]), format="%d/%m/%Y %H:%M")

    df_horas_aula["DIA-SEMANA"] = df_horas_aula["DATA"].dt.weekday
    df_horas_aula["MES"] = df_horas_aula["DATA"].dt.month

    df_agua.update(df_agua.drop(["DATA"], axis=1).astype(int))
    df_energia.update(df_energia.drop(["DATA"], axis=1).astype(int))
    df_horas_aula.update(df_horas_aula.drop("DATA", axis=1).astype(int))
    df_clima.update(df_clima[["PRECIPITACAO"]].astype(int))
    for coluna in df_clima.drop(["DATA", "PRECIPITACAO"], axis=1):
        df_clima[coluna] = df_clima[coluna].astype("float32")

    return df_agua, df_energia, df_clima, df_horas_aula


def agrupamentoDiarioMedia(df):
    df = df.groupby(pd.Grouper(key="DATA", freq="D")).mean().reset_index()
    return df


def agrupamentoMensal(df, datas=None, strategy="mean"):
    test = 1
    if "DIA-SEMANA" in df.columns:
        dfDias = pd.get_dummies(df["DIA-SEMANA"].astype(int), prefix="", prefix_sep="", dtype=int).rename(
            columns={"0": 'SEG', "1": 'TER', "2": 'QUA', "3": 'QUI', "4": 'SEX', "5": 'SAB', "6": 'DOM'})
        df = pd.concat([df, dfDias], axis=1)
        df = df.drop(["DIA-SEMANA"], axis=1)
    if "DIA" in df.columns:
        df = df.drop(["DIA"], axis=1)
    if "HORA" in df.columns:
        df = df.drop(["HORA"], axis=1)
    if "MES" in df.columns:
        df = df.groupby(pd.cut(df["DATA"], right=True, bins=datas,
                               labels=pd.to_datetime(datas[1:].tolist())), observed=False).agg(
            {**{col: "last" for col in df[["DATA", "MES"]].columns},
             **{col: strategy for col in df.drop(["DATA", "MES"], axis=1).columns}}).reset_index(drop=True)
    elif datas is not None and np.any(datas):
        df = df.groupby(pd.cut(df["DATA"], right=True, bins=datas,
                               labels=pd.to_datetime(datas[1:].tolist())), observed=False).agg(
            {**{col: "last" for col in df[["DATA"]].columns},
             **{col: strategy for col in df.drop(["DATA"], axis=1).columns}}).reset_index(drop=True)
    else:
        df = df.groupby(pd.Grouper(key="DATA", freq="M")).mean()
    if "MES" in df.columns:
        dfMeses = pd.get_dummies(df["MES"].astype(int), prefix="", prefix_sep="", dtype=int).rename(
            columns={"1": "JAN", "2": "FEV", "3": "MAR", "4": "ABR", "5": "MAI", "6": "JUN", "7": "JUL", "8": "AGO",
                     "9": "SET", "10": "OUT", "11": "NOV", "12": "DEZ"}
        )
        df = pd.concat([df, dfMeses], axis=1)
        df = df.drop(["MES"], axis=1)

    return df


def obterLags(df, var=None, lags=3):
    dfCopy = df.copy()
    if var is not None:
        colunas = [var]
    else:
        colunas = dfCopy.columns.values

    for coluna in colunas:
        if coluna in ["SEG", "TER", "QUA", "QUI", "SEX", "SAB", "DOM"]:
            continue
        for i in range(1, lags + 1):
            lag = dfCopy[coluna].shift(i)
            dfCopy[f'{var}_LAG_' + "{:02d}".format(i)] = lag
    return dfCopy.sort_index(axis=1)


def tratamentoNulosDropLinhas(df):
    return df.dropna()


def tratamentoNulosDropColunas(df):
    return df.dropna(axis=1)


def tratamentoNulosMediaColuna(df):
    return df.fillna(df.mean())


def tratamentoNulosMediaSupInf(df):
    return df.apply(lambda x: x.fillna((x.ffill() + x.bfill()) / 2))
