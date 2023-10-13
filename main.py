import pandas as pd

from analise import analiseCorrelacao
from preprocessamento import prepararDados, obterLags, agrupamentoDiarioMedia, agrupamentoMensalMedia, tratamentoNulos
from treinoTeste import plotTreinoTeste, treinarRF, testar


def treinamentoAgua(dfAgua):
    modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfAgua, ["CONSUMO (M3)"])
    yPrevisto = testar(modelo, xTeste, yTeste)
    dfResultado = pd.DataFrame(xTeste)
    dfResultado["PREVISÃO CONSUMO (M3)"] = yPrevisto
    dfResultado["ESPERADO"] = yTeste

    dfResultado = dfResultado.sort_index()

    plotTreinoTeste(dfResultado.index.values, dfResultado["PREVISÃO CONSUMO (M3)"], dfResultado["ESPERADO"],
                    title="CONSUMO (M3)")


def treinamentoEnergia(dfEnergia):
    modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfEnergia, ["ENERGIA ELÉTRICA PONTA (KWh)",
                                                                     "ENERGIA ELÉTRICA FORA DA PONTA (KWh)"])
    yPrevisto = testar(modelo, xTeste, yTeste)
    dfPrevisto = pd.DataFrame(yPrevisto)
    dfResultado = pd.DataFrame(xTeste)
    dfResultado["ENERGIA ELÉTRICA PONTA (KWh)"] = yTeste["ENERGIA ELÉTRICA PONTA (KWh)"]
    dfResultado["ENERGIA ELÉTRICA FORA DA PONTA (KWh)"] = yTeste["ENERGIA ELÉTRICA FORA DA PONTA (KWh)"]
    dfResultado["PREVISÃO ENERGIA ELÉTRICA PONTA (KWh)"] = dfPrevisto.iloc[:, 0].values
    dfResultado["PREVISÃO ENERGIA ELÉTRICA FORA DA PONTA (KWh)"] = dfPrevisto.iloc[:, 1].values

    dfResultado = dfResultado.sort_index()

    plotTreinoTeste(dfResultado.index.values,
                    dfResultado["PREVISÃO ENERGIA ELÉTRICA PONTA (KWh)"],
                    dfResultado["ENERGIA ELÉTRICA PONTA (KWh)"])

    plotTreinoTeste(dfResultado.index.values,
                    dfResultado["PREVISÃO ENERGIA ELÉTRICA FORA DA PONTA (KWh)"],
                    dfResultado["ENERGIA ELÉTRICA FORA DA PONTA (KWh)"])


dfAgua, dfEnergia, dfClima, dfHorasAula = prepararDados()


dfGroup = agrupamentoMensalMedia(dfHorasAula, datas=dfEnergia.index.values)
print(dfGroup)