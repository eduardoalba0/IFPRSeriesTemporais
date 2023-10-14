import sys

import pandas as pd

from preprocessamento import prepararDados, obterLags, agrupamentoDiarioMedia, agrupamentoMensalMedia, tratamentoNulos
from treinoTeste import plotTreinoTeste, treinarRF, testar


def treinoTeste(dfAgua):
    modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfAgua, ["CONSUMO (M3)"])
    x, yEsperado, yPrevisto = testar(modelo, xTeste, yTeste)
    dfResultado = pd.DataFrame(x)
    dfResultado["CONSUMO ESPERADO (M3)"] = yEsperado
    dfResultado["CONSUMO PREVISTO (M3)"] = yPrevisto
    dfResultado = dfResultado.sort_index()

    plotTreinoTeste(dfResultado.index.values, dfResultado["CONSUMO PREVISTO (M3)"],
                    dfResultado["CONSUMO ESPERADO (M3)"],
                    title="CONSUMO (M3)")


def treinamentoEnergia(dfEnergia):
    modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfEnergia, ["CONSUMO (KWh)"])
    x, yEsperado, yPrevisto = testar(modelo, xTeste, yTeste)
    dfPrevisto = pd.DataFrame(yPrevisto)
    dfResultado = pd.DataFrame(x)
    dfResultado["CONSUMO PREVISTO (KWh)"] = dfPrevisto.iloc[:, 0].values
    dfResultado["CONSUMO ESPERADO (KWh)"] = yEsperado["CONSUMO (KWh)"]
    dfResultado = dfResultado.sort_index()

    plotTreinoTeste(dfResultado.index.values,
                    dfResultado["CONSUMO PREVISTO (KWh)"],
                    dfResultado["CONSUMO ESPERADO (KWh)"])


def main(argv):
    dfAgua, dfEnergia, dfClima, dfHorasAula = prepararDados()
    dfHorasAula = agrupamentoMensalMedia(dfHorasAula, datas=dfAgua.index.values)
    dfClima = agrupamentoMensalMedia(dfClima, datas=dfAgua.index.values)
    dfAgua = dfAgua.iloc[1:, 2:]

    dfMerged = dfAgua.merge(dfHorasAula, right_index=True, left_index=True, how="inner")
    dfMerged = dfMerged.merge(dfClima, right_index=True, left_index=True, how="inner")
    dfMerged = obterLags(dfMerged, ["CONSUMO (M3)"], lags=4)

    treinoTeste(dfMerged)


if __name__ == '__main__':
    main(sys.argv)
