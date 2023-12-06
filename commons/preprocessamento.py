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

    df_agua = df_agua.set_index("DATA")
    df_energia = df_energia.set_index("DATA")

    df_clima["DIA-SEMANA"] = df_clima["DATA"].dt.weekday

    df_agua = df_agua.astype(int)
    df_energia = df_energia.astype(int)

    df_clima = df_clima.set_index("DATA")
    df_horas_aula = df_horas_aula.set_index("DATA")

    df_clima.update(df_clima[["PRECIPITACAO","DIA-SEMANA"]].astype(int))
    for coluna in df_clima.drop(["PRECIPITACAO","DIA-SEMANA"], axis=1):
        df_clima[coluna] = df_clima[coluna].astype(float)

    df_horas_aula = df_horas_aula.astype(int)

    return df_agua, df_energia, df_clima, df_horas_aula


def agrupamentoDiarioMedia(df):
    df = df.reset_index()
    df = df.groupby(pd.Grouper(key="DATA", freq="D")).mean()
    return df


def agrupamentoMensal(df, datas=None, strategy="mean"):
    if "DIA-SEMANA" in df.columns:
        dfDias = pd.get_dummies(df["DIA-SEMANA"], prefix="", prefix_sep="", dtype=int).rename(
            columns={"0": 'SEG', "1": 'TER', "2": 'QUA', "3": 'QUI', "4": 'SEX', "5": 'SAB', "6": 'DOM'})
        df = pd.concat([df, dfDias], axis=1)
        df = df.drop(["DIA-SEMANA"], axis=1)
    if "DIA" in df.columns:
        df = df.drop(["DIA"], axis=1)
    if "HORA" in df.columns:
        df = df.drop(["HORA"], axis=1)
    if datas is not None and np.any(datas):
        df = df.groupby(pd.cut(df.index, right=True, bins=datas,
                               labels=pd.to_datetime(datas[1:].tolist())), observed=False).agg(
            {**{col: "last" for col in df.columns},
             **{col: strategy for col in df.columns}})
    else:
        df = df.reset_index()
        df = df.groupby(pd.Grouper(key="DATA", freq="M")).mean()
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
