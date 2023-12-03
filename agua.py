import time

from commons.exploracao import plotTreinoTeste
from commons.ga_rf import GARF
from commons.ga_svr import GASVR
from commons.preprocessamento import prepararDados, agrupamentoMensalMedia, obterLags
from commons.treinoTeste import treinarRF, treinarSVR

if __name__ == '__main__':
    dfAgua, dfEnergia, dfClima, dfHorasAula = prepararDados()

    print("------------------------ INICIANDO TESTE DO CONSUMO DE ÁGUA ------------------------ ")

    variavel = "CONSUMO"

    dfHorasAula = agrupamentoMensalMedia(dfHorasAula, datas=dfAgua.index.values)
    dfClima = agrupamentoMensalMedia(dfClima, datas=dfAgua.index.values)
    dfAgua = dfAgua.iloc[1:, :]

    dfMerged = dfAgua.merge(dfHorasAula, right_index=True, left_index=True, how="inner")
    dfMerged = dfMerged.merge(dfClima, right_index=True, left_index=True, how="inner")

    t_inicio = time.time()
    #ga = GARF(dfMerged, variavel, 100, 100, 0.5, 5, 1234)
    ga = GASVR(dfMerged, variavel, 100, 100, 0.5, 5, 1234)
    populacao = ga.run()
    best = populacao[0]
    t_fim = time.time()
    print(f"Tempo de execução: {t_fim - t_inicio} segundos")

    #modelo, dfResultado, dfResumo = treinarRF(dfMerged, variavel, estimators=best.n_estimators,
    #                                                     maxDepth=best.max_depth, minSampleLeaf=best.min_sample_leaf,
    #                                                     nLags=best.n_lags, folds=12)

    modelo, dfResultado, dfResumo = treinarSVR(dfMerged, variavel, kernel=best.kernel,
                                               epsilon=best.epsilon, gamma=best.gamma, c=best.c,
                                               nLags=best.n_lags, folds=12)

    plotTreinoTeste(dfResultado.index.values, dfResultado["PREVISTO"],
                        dfResultado["ESPERADO"], title="CONSUMO DE ÁGUA")
    print(dfResumo.head())