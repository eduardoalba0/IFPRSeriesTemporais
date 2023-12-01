from commons.exploracao import plotTreinoTeste
from commons.preprocessamento import prepararDados, agrupamentoMensalMedia, obterLags
from commons.treinoTeste import treinarRF, testar

dfAgua, dfEnergia, dfClima, dfHorasAula = prepararDados()
variavel = "CONSUMO (M3)"

print("------------------------ INICIANDO TESTE DO CONSUMO DE ÁGUA ------------------------ ")
dfHorasAula = agrupamentoMensalMedia(dfHorasAula, datas=dfAgua.index.values)
dfClima = agrupamentoMensalMedia(dfClima, datas=dfAgua.index.values)
dfAgua = dfAgua.iloc[1:, :]

dfMerged = dfAgua
dfMerged = dfAgua.merge(dfHorasAula, right_index=True, left_index=True, how="inner")
dfMerged = dfMerged.merge(dfClima, right_index=True, left_index=True, how="inner")

modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfMerged, variavel, nLags=6)

print("Conjunto de Treinamento: %d" % len(xTreino))
print("Conjunto de Teste: %d" % len(xTeste))

dfResultado, dfResumo = testar(modelo, xTeste, yTeste)

for variavel in variavel:
    plotTreinoTeste(dfResultado.index.values, dfResultado[f"{variavel} - PREVISTO"],
                    dfResultado[f"{variavel} - ESPERADO"], title="CONSUMO DE ÁGUA")
print(dfResumo.head())