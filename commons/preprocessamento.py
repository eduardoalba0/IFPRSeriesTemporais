import numpy as np
import pandas as pd

def prepararDados():
    dfAgua = pd.read_csv("./Dados/ConsumoAguaIFPR.csv", delimiter=";")
    dfEnergia = pd.read_csv("./Dados/ConsumoEnergiaIFPR.csv", delimiter=";")
    dfClima = pd.read_csv("./Dados/ClimaAgritempo.csv", delimiter=";")
    dfHorasAula = pd.read_csv("./Dados/HorasAulaIFPR.csv", delimiter=";")
    dfAgua["DATA"] = pd.to_datetime(dfAgua["DATA"], format="%d/%m/%Y")
    dfEnergia["DATA"] = pd.to_datetime(dfEnergia["DATA"], format="%d/%m/%Y")
    dfClima["DATA"] = pd.to_datetime((dfClima["DATA"]), format="%d/%m/%Y")
    dfHorasAula["DATA"] = pd.to_datetime((dfHorasAula["DATA"]), format="%d/%m/%Y %H:%M")

    dfAgua["ANO"] = dfAgua["DATA"].dt.year
    dfAgua["MES"] = dfAgua["DATA"].dt.month
    dfAgua = dfAgua.set_index("DATA")
    dfAgua = dfAgua.astype(int)

    dfEnergia["ANO"] = dfEnergia["DATA"].dt.year
    dfEnergia["MES"] = dfEnergia["DATA"].dt.month
    dfEnergia = dfEnergia.set_index("DATA")
    dfEnergia = dfEnergia.astype(int)

    dfClima["ANO"] = dfClima["DATA"].dt.year
    dfClima["MES"] = dfClima["DATA"].dt.month
    dfClima["DIA"] = dfClima["DATA"].dt.day
    dfClima["DIA-SEMANA"] = dfClima["DATA"].dt.weekday
    dfClima = dfClima.set_index("DATA")
    dfClima.update(dfClima[["ANO", "MES", "DIA", "DIA-SEMANA"]].astype(int))
    for coluna in dfClima.drop(["ANO", "MES", "DIA", "DIA-SEMANA"], axis=1):
        dfClima[coluna] = dfClima[coluna].astype(float)

    dfHorasAula["ANO"] = dfHorasAula["DATA"].dt.year
    dfHorasAula["MES"] = dfHorasAula["DATA"].dt.month
    dfHorasAula["DIA"] = dfHorasAula["DATA"].dt.day
    dfHorasAula["DIA-SEMANA"] = dfHorasAula["DATA"].dt.weekday
    dfHorasAula["HORA"] = dfHorasAula["DATA"].dt.hour
    dfHorasAula = dfHorasAula.set_index("DATA")
    dfHorasAula = dfHorasAula.astype(int)

    return dfAgua, dfEnergia, dfClima, dfHorasAula

def agrupamentoDiarioMedia(df):
    df = df.reset_index()
    df = df.drop("HORA", axis=1)
    df = df.groupby(pd.Grouper(key="DATA", freq="D")).mean()
    return df


def agrupamentoDiarioSoma(df):
    df = df.reset_index()
    df = df.drop("HORA", axis=1)
    df = df.groupby(pd.Grouper(key="DATA", freq="D")).sum()
    return df


def agrupamentoMensalMedia(df, datas=None):
    if "DIA-SEMANA" in df.columns:
        dfDias = pd.get_dummies(df["DIA-SEMANA"], prefix="", prefix_sep="", dtype=int).rename(
            columns={"0": 'SEG', "1": 'TER', "2": 'QUA', "3": 'QUI', "4": 'SEX', "5": 'SAB', "6": 'DOM'})
        df = pd.concat([df, dfDias], axis=1)
        df = df.drop(["DIA", "DIA-SEMANA"], axis=1)
    if "DIA" in df.columns:
        df = df.drop(["DIA"], axis=1)
    if "HORA" in df.columns:
        df = df.drop(["HORA"], axis=1)
    if datas is not None and np.any(datas):
        df = df.groupby(pd.cut(df.index, right=True, bins=datas,
                               labels=pd.to_datetime(datas[1:].tolist())), observed=False).agg(
            {**{col: "last" for col in df[["MES", "ANO"]].columns},
             **{col: "mean" for col in df.drop(["MES", "ANO"], axis=1).columns}})
    else:
        df = df.reset_index()
        df = df.groupby(pd.Grouper(key="DATA", freq="M")).mean()
    return df


def agrupamentoMensalSoma(df, datas=None):
    if "DIA-SEMANA" in df.columns:
        dfDias = pd.get_dummies(df["DIA-SEMANA"], prefix="", prefix_sep="", dtype=int).rename(
            columns={"0": 'SEG', "1": 'TER', "2": 'QUA', "3": 'QUI', "4": 'SEX', "5": 'SAB', "6": 'DOM'})
        df = pd.concat([df, dfDias], axis=1)
        df = df.drop(["DIA", "DIA-SEMANA"], axis=1)
    if "DIA" in df.columns:
        df = df.drop(["DIA"], axis=1)
    if "HORA" in df.columns:
        df = df.drop(["HORA"], axis=1)
    if datas is not None and np.any(datas):
        df = df.groupby(pd.cut(df.index, right=True, bins=datas,
                               labels=pd.to_datetime(datas[1:].tolist())), observed=False).agg(
            {**{col: "last" for col in df[["MES", "ANO"]].columns},
             **{col: "sum" for col in df.drop(["MES", "ANO"], axis=1).columns}})
    else:
        df = df.reset_index()
        df = df.groupby(pd.Grouper(key="DATA", freq="M")).sum()
    return df


def obterLags(df, var=None, lags=3):
    if var is not None:
        colunas = [var]
    else:
        colunas = df.columns.values

    for coluna in colunas:
        if coluna in ["SEG", "TER", "QUA", "QUI", "SEX", "SAB", "DOM"]:
            continue
        for i in range(1, lags + 1):
            lag = df[coluna].shift(i)
            df[f'{coluna}_LAG_{i}'] = lag
    return df.sort_index(axis=1)


def tratamentoNulosDropLinhas(df):
    return df.dropna()


def tratamentoNulosDropColunas(df):
    return df.dropna(axis=1)


def tratamentoNulosMediaColuna(df):
    return df.fillna(df.mean())


def tratamentoNulosMediaSupInf(df):
    return df.apply(lambda x: x.fillna((x.ffill() + x.bfill()) / 2))