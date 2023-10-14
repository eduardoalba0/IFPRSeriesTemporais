import numpy as np
import pandas as pd

from treinoTeste import treinarRF, plotTreinoTeste, testar


def prepararDados():
    dfAgua = pd.read_csv("./Dados/ConsumoAguaIFPR.csv", delimiter=";")
    dfEnergia = pd.read_csv("./Dados/ConsumoEnergiaIFPR.csv", delimiter=";")
    dfClima = pd.read_csv("./Dados/ClimaINMETClevelandia.csv", delimiter=";")
    dfHorasAula = pd.read_csv("./Dados/HorasAulaIFPR.csv", delimiter=";")
    dfAgua["DATA"] = pd.to_datetime(dfAgua["DATA"], format="%d/%m/%Y")
    dfEnergia["DATA"] = pd.to_datetime(dfEnergia["DATA"], format="%d/%m/%Y")
    dfClima["DATA"] = pd.to_datetime((dfClima["DATA"]), format="%d/%m/%Y %H:%M")
    dfHorasAula["DATA"] = pd.to_datetime((dfHorasAula["DATA"]), format="%d/%m/%Y %H:%M")

    dfAgua = dfAgua.set_index("DATA")
    for coluna in dfAgua.columns:
        dfAgua[coluna] = dfAgua[coluna].astype(int)

    dfEnergia = dfEnergia.set_index("DATA")
    for coluna in dfEnergia.columns:
        dfEnergia[coluna] = dfEnergia[coluna].astype(int)
    dfEnergia["CONSUMO (KWh)"] = dfEnergia["ENERGIA ELÉTRICA PONTA (KWh)"] + dfEnergia[
        "ENERGIA ELÉTRICA FORA DA PONTA (KWh)"] + dfEnergia["ENERGIA REATIVA PONTA (KWh)"] + dfEnergia[
                                     "ENERGIA REATIVA FORA DA PONTA (KWh)"]
    dfEnergia = dfEnergia.drop(
        ["ENERGIA ELÉTRICA PONTA (KWh)", "ENERGIA ELÉTRICA FORA DA PONTA (KWh)", "ENERGIA REATIVA PONTA (KWh)",
         "ENERGIA REATIVA FORA DA PONTA (KWh)"], axis=1)

    dfClima = dfClima.set_index("DATA")
    for coluna in dfClima.columns[:5]:
        dfClima[coluna] = dfClima[coluna].astype(int)
    for coluna in dfClima.columns[5:]:
        dfClima[coluna] = dfClima[coluna].astype(float)

    dfHorasAula = dfHorasAula.set_index("DATA")
    for coluna in dfHorasAula.columns[0:]:
        dfHorasAula[coluna] = dfHorasAula[coluna].astype(int)

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
    df = df.drop(["DIA", "DIA-SEMANA", "HORA"], axis=1)
    if datas is not None and np.any(datas):
        df = df.groupby(pd.cut(df.index, right=True, bins=datas,
                               labels=pd.to_datetime(datas[1:].tolist())), observed=False).agg(
            {**{col: "last" for col in df.columns[:2]}, **{col: "mean" for col in df.columns[2:]}})
    else:
        df = df.reset_index()
        df = df.groupby(pd.Grouper(key="DATA", freq="M")).mean()
    return df

def agrupamentoMensalSoma(df, datas=None):
    df = df.drop(["DIA", "DIA-SEMANA", "HORA"], axis=1)
    if datas is not None and np.any(datas):
        df = df.groupby(pd.cut(df.index, right=True, bins=datas,
                               labels=pd.to_datetime(datas[1:].tolist())), observed=False).agg(
            {**{col: "last" for col in df.columns[:2]}, **{col: "sum" for col in df.columns[2:]}})
    else:
        df = df.reset_index()
        df = df.groupby(pd.Grouper(key="DATA", freq="M")).sum()
    return df


def obterLags(df, vars=None, lags=3):
    if vars is not None:
        colunas = vars
    else:
        colunas = df.columns.values

    for coluna in colunas:
        for i in range(1, lags + 1):
            df[f'{coluna}_LAG_{i}'] = df[coluna].shift(i)
    return df.sort_index(axis=1)


def tratamentoNulos(df, matrizCorrelacao=None):
    dfAux = df.copy()
    dfAux = df.dropna(thresh=dfAux.shape[1] - 2)
    for coluna in dfAux.columns[dfAux.isnull().any()]:
        if coluna == "MÊS" or "_LAG_" in coluna:
            continue
        if matrizCorrelacao is not None and np.any(matrizCorrelacao):
            cols = matrizCorrelacao[coluna][matrizCorrelacao[coluna] > 0.4].index
            if len(cols) <= 1:
                continue
        modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfAux[cols], [coluna])
        yPrevisto = testar(modelo, xTeste, yTeste)
        dfResultado = pd.DataFrame(xTeste)
        dfResultado["PREVISÃO"] = yPrevisto
        dfResultado["ESPERADO"] = yTeste

        dfResultado = dfResultado.sort_index()
        plotTreinoTeste(dfResultado.index.values, dfResultado["PREVISÃO"], dfResultado["ESPERADO"], title=coluna)
