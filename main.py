import threading
import time

from commons.exploracao import plotTreinoTeste, plotPrevisao, plotHistResiduos
from commons.ga_rf import GARF
from commons.ga_svr import GASVR
from commons.preprocessamento import prepararDados, agrupamentoMensal, agrupamentoDiarioMedia
from commons.previsao import prever
from commons.treinoTeste import treinarRF, treinarSVR

if __name__ == '__main__':
    dfAgua, dfEnergia, dfClima, dfHorasAula = prepararDados()

    h_previsoes = 6
    individuos = 1
    geracoes = 1

    print("------------------------ INICIANDO TESTE DO CONSUMO DE ENERGIA ------------------------ ")

    var = "CONSUMO"
    df = dfEnergia

    dfHorasAula = agrupamentoMensal(agrupamentoDiarioMedia(dfHorasAula), datas=df.index.values, strategy="sum")
    dfClima = agrupamentoMensal(dfClima, datas=df.index.values, strategy="sum")
    df = df.iloc[1:, :]

    dfMerged = df.merge(dfHorasAula, right_index=True, left_index=True, how="inner")
    dfMerged = dfMerged.merge(dfClima, right_index=True, left_index=True, how="inner")

    print(f"População com {individuos} individuos e {geracoes} gerações.")
    t_inicio = time.time()
    ga = GARF(dfMerged, var, individuos, geracoes, 0.5, 5, 1234)
    # ga = GASVR(dfMerged, var, individuos, geracoes, 0.5, 4, 1234)
    populacao = ga.run()
    t_fim = time.time()
    best = populacao[0]
    print(best)
    print(f"Tempo de execução: {t_fim - t_inicio} segundos")

    print("Random Forest")
    modelo, dfResultado, dfResumo = treinarRF(dfMerged, var, estimators=best.n_estimators,
                                              maxDepth=best.max_depth, nLags=best.n_lags, folds=h_previsoes,
                                              semente=ga.semente)

    # print("SVR")
    # modelo, dfResultado, dfResumo = treinarSVR(dfMerged, var, kernel=best.kernel,
    #                                            epsilon=best.epsilon, c=best.c,
    #                                            nLags=best.n_lags, folds=h_previsoes)
    plotTreinoTeste(dfResultado["PREVISTO"],
                    dfResultado["OBSERVADO"], title=f"CONSUMO DE ENERGIA {h_previsoes} PASSOS À FRENTE")
    plotHistResiduos(dfResultado["PREVISTO"].tail(h_previsoes),
                    dfResultado["OBSERVADO"].tail(h_previsoes), title=f"HISTOGRAMA DE RESÍDUOS - CONSUMO DE ENERGIA {h_previsoes} PASSOS À FRENTE")
    print(dfResumo.head())

    dfPrevisao = prever(modelo, dfMerged, var, h_previsoes, best.n_lags)

    plotPrevisao(dfPrevisao["PREVISTO"], dfPrevisao["OBSERVADO"], title=f"PREVISÃO DO CONSUMO DE ENERGIA {h_previsoes} PASSOS À FRENTE")

