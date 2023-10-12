import numpy as np
import pandas as pd

from treinoTeste import treinarRF, plotTreinoTeste, testar


def prepararDados():
    dfAgua = pd.read_csv("./Dados/ConsumoAguaIFPR.csv", delimiter=";")
    dfEnergia = pd.read_csv("./Dados/ConsumoEnergiaIFPR.csv", delimiter=";")
    dfClima = pd.read_csv("./Dados/ClimaINMETClevelandia.csv", delimiter=";")
    dfAgua["DATA LEITURA"] = pd.to_datetime(dfAgua["DATA LEITURA"], format="%d/%m/%Y")
    dfEnergia["DATA LEITURA"] = pd.to_datetime(dfEnergia["DATA LEITURA"], format="%d/%m/%Y")
    dfClima["DATA"] = pd.to_datetime((dfClima["DATA"] + " " + dfClima["HORA"]),
                                     format="%d/%m/%Y %H:%M")

    dfAgua.set_index("DATA LEITURA", inplace=True)
    dfEnergia.set_index("DATA LEITURA", inplace=True)
    dfClima.set_index("DATA", inplace=True)

    for coluna in dfAgua.columns:
        dfAgua[coluna] = dfAgua[coluna].astype(float)

    for coluna in dfEnergia.columns:
        dfEnergia[coluna] = dfEnergia[coluna].astype(float)

    dfClima.dropna(inplace=True, how="all")
    dfClima.drop("HORA", axis=1, inplace=True)
    for coluna in dfClima.columns[1:]:
        dfClima[coluna] = dfClima[coluna].astype(float)

    return dfAgua, dfEnergia, dfClima


def agrupamentoDiarioMedia(df):
    df = df.groupby(pd.Grouper(key="DATA", freq="D")).mean()
    return df

def agrupamentoMensalMedia(df):
    df = df.groupby(pd.Grouper(key="DATA", freq="M")).mean()
    return df


def obterLags(df):
    for coluna in df.columns.values:
        for i in range(1, 3 + 1):
            df[f'{coluna}_LAG_{i}'] = df[coluna].shift(i)
    return df.sort_index(axis=1)


def tratamentoNulos(df, matrizCorrelacao = None):
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

        dfResultado.sort_index(inplace=True)
        plotTreinoTeste(dfResultado.index.values, dfResultado["PREVISÃO"], dfResultado["ESPERADO"], title=coluna)
