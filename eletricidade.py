import numpy as np

from commons.exploracao import plotTreinoTeste, acf
from commons.preprocessamento import prepararDados, agrupamentoMensalMedia
from commons.treinoTeste import treinarRF, testar

dfAgua, dfEnergia, dfClima, dfHorasAula = prepararDados()

print("------------------------ INICIANDO TESTE DO CONSUMO DE ENERGIA ------------------------ ")

variavel = "CONSUMO (KWh)"

dfHorasAula = agrupamentoMensalMedia(dfHorasAula, datas=dfEnergia.index.values)
dfClima = agrupamentoMensalMedia(dfClima, datas=dfEnergia.index.values)
dfEnergia = dfEnergia.iloc[1:, :]

dfMerged = dfEnergia.merge(dfHorasAula, right_index=True, left_index=True, how="inner")
dfMerged = dfMerged.merge(dfClima, right_index=True, left_index=True, how="inner")

modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfMerged, variavel, nLags=3)

print("Conjunto de Treinamento: %d" % len(xTreino))
print("Conjunto de Teste: %d" % len(xTeste))

dfResultado, dfResumo = testar(modelo, xTeste, yTeste)

plotTreinoTeste(dfResultado.index.values, dfResultado["PREVISTO"],
dfResultado["ESPERADO"], title="CONSUMO DE ENERGIA - RF")
print(dfResumo.head())
