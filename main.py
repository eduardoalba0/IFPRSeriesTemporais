import sys

import pandas as pd

from preprocessamento import prepararDados, obterLags, agrupamentoDiarioMedia, agrupamentoMensalMedia, tratamentoNulos, \
    agrupamentoMensalSoma
from treinoTeste import plotTreinoTeste, treinarRF, testar


def treinoTesteAgua():
    dfAgua, dfEnergia, dfClima, dfHorasAula = prepararDados()
    dfHorasAula = agrupamentoMensalSoma(dfHorasAula, datas=dfAgua.index.values)
    dfClima = agrupamentoMensalMedia(dfClima, datas=dfAgua.index.values)
    dfAgua = dfAgua.iloc[1:, :]
    dfHorasAula = dfHorasAula.drop(["ANO", "MES"], axis=1)
    dfClima = dfClima.drop(["ANO", "MES"], axis=1)

    dfMerged = dfAgua.merge(dfHorasAula, right_index=True, left_index=True, how="inner")
    dfMerged = dfMerged.merge(dfClima, right_index=True, left_index=True, how="inner")
    dfMerged = obterLags(dfMerged, lags=1)

    modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfMerged, ["CONSUMO (M3)"])
    x, yEsperado, yPrevisto = testar(modelo, xTeste, yTeste)
    dfResultado = pd.DataFrame(x)
    dfResultado["CONSUMO ESPERADO (M3)"] = yEsperado
    dfResultado["CONSUMO PREVISTO (M3)"] = yPrevisto
    dfResultado = dfResultado.sort_index()

    plotTreinoTeste(dfResultado.index.values, dfResultado["CONSUMO PREVISTO (M3)"],
                    dfResultado["CONSUMO ESPERADO (M3)"],
                    title="CONSUMO (M3)")


def treinoTesteEnergia():
    dfAgua, dfEnergia, dfClima, dfHorasAula = prepararDados()
    dfHorasAula = agrupamentoMensalSoma(dfHorasAula, datas=dfEnergia.index.values)
    dfClima = agrupamentoMensalMedia(dfClima, datas=dfEnergia.index.values)
    dfEnergia = dfEnergia.iloc[1:, :]
    dfHorasAula = dfHorasAula.drop(["ANO", "MES"], axis=1)
    dfClima = dfClima.drop(["ANO", "MES"], axis=1)
    dfClima = dfClima.apply(lambda x: x.fillna((x.ffill() + x.bfill()) / 2))

    dfMerged = dfEnergia.merge(dfHorasAula, right_index=True, left_index=True, how="inner")
    dfMerged = dfMerged.merge(dfClima, right_index=True, left_index=True, how="inner")
    dfMerged = obterLags(dfMerged, ["CONSUMO (KWh)"], lags=1)

    modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfMerged, ["CONSUMO (KWh)"])
    x, yEsperado, yPrevisto = testar(modelo, xTeste, yTeste)
    dfResultado = pd.DataFrame(x)
    dfResultado["CONSUMO ESPERADO (KWh)"] = yEsperado
    dfResultado["CONSUMO PREVISTO (KWh)"] = yPrevisto
    dfResultado = dfResultado.sort_index()

    plotTreinoTeste(dfResultado.index.values,
                    dfResultado["CONSUMO PREVISTO (KWh)"],
                    dfResultado["CONSUMO ESPERADO (KWh)"])


def main(argv):
    treinoTesteAgua()


if __name__ == '__main__':
    main(sys.argv)
