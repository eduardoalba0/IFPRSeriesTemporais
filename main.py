import threading
import time

from commons.exploracao import plotTreinoTeste, plotPrevisao, plotHistResiduos
from commons.ga_rf import GARF, IndividuoRF
from commons.ga_svr import GASVR, IndividuoSVR
from commons.preprocessamento import prepararDados, agrupamentoMensal, agrupamentoDiarioMedia
from commons.previsao import prever
from commons.treinoTeste import treinarSVR, treinarRF

if __name__ == '__main__':
    dfAgua, dfEnergia, dfClima, dfHorasAula = prepararDados()

    var = "CONSUMO"

    h_previsoes = 3
    individuos = 2
    geracoes = 10
    algoritmo = "SVR"  # SVR ou RF
    previsao = "água"  # água ou energia
    clima = True  # True ou False
    otimizar = True  # True ou False

    # best =

    if previsao == "água":
        print("------------------------ INICIANDO TESTE DO CONSUMO DE ÁGUA ------------------------ ")
        df = dfAgua
    elif previsao == "energia":
        print("------------------------ INICIANDO TESTE DO CONSUMO DE ENERGIA ELÉTRICA ------------------------ ")
        df = dfEnergia

    dfHorasAula = agrupamentoMensal(agrupamentoDiarioMedia(dfHorasAula), datas=df.index.values, strategy="sum")
    dfClima = agrupamentoMensal(dfClima, datas=df.index.values, strategy="sum")
    df = df.iloc[1:, :]

    dfMerged = df.merge(dfHorasAula, right_index=True, left_index=True, how="inner")

    if clima:
        dfMerged = dfMerged.merge(dfClima, right_index=True, left_index=True, how="inner")

    print(f"População com {individuos} individuos e {geracoes} gerações.")

    if otimizar:
        if algoritmo == "SVR":
            print("SVR")
            t_inicio = time.time()
            ga = GASVR(dfMerged, var, individuos, geracoes, 0.5, h_previsoes, (12 * individuos * geracoes))
            populacao = ga.run()
            t_fim = time.time()
        elif algoritmo == "RF":
            print("Random Forest")
            ga = GARF(dfMerged, var, individuos, geracoes, 0.5, h_previsoes, (12 * individuos * geracoes))
            populacao = ga.run()
            t_fim = time.time()
        best = populacao[0]
        print(best)
        print(f"Tempo de execução: {t_fim - t_inicio} segundos")

    if algoritmo == "SVR":
        modelo, dfResultado, dfResumo = treinarSVR(dfMerged, var, kernel=best.kernel,
                                                   epsilon=best.epsilon, c=best.c,
                                                   nLags=best.n_lags, folds=h_previsoes)
    elif algoritmo == "RF":
        modelo, dfResultado, dfResumo = treinarRF(dfMerged, var, estimators=best.n_estimators,
                                                  maxDepth=best.max_depth, nLags=best.n_lags, folds=h_previsoes,
                                                  semente=(12 * individuos * geracoes))

    plotTreinoTeste(dfResultado["PREVISTO"],
                    dfResultado["OBSERVADO"],
                    title=f"Consumo de {previsao} {h_previsoes} passos à frente - Treino/Teste {algoritmo}")
    plotHistResiduos(dfResultado["PREVISTO"].tail(h_previsoes),
                     dfResultado["OBSERVADO"].tail(h_previsoes),
                     title=f"Histograma de Resíduos da Previsão do Consumo de {previsao} {h_previsoes} passos à frente - Algoritmo {algoritmo}")
    print(dfResumo.head())

    dfPrevisao = prever(modelo, dfMerged, var, h_previsoes, best.n_lags)

    plotPrevisao(dfPrevisao["PREVISTO"], dfPrevisao["OBSERVADO"],
                 title=f"PRevisão do Consumo de {previsao} {h_previsoes} passos à frente - Algoritmo {algoritmo}")
