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

    h_previsoes = 12

    individuos = 500
    geracoes = 1000
    algoritmo = "RF"  # SVR ou RF
    previsao = "energia"  # água ou energia
    clima = True  # True ou False
    otimizar = False  # True ou False

    # bestSVR_A_Clima = IndividuoSVR().create(2, "poly", 0, 4373) #MELHOR H-6 AGUA //
    # bestSVR_A_SClima = IndividuoSVR().create(1, "poly", 0, 3771)
    #
    # bestSVR_E_Clima = IndividuoSVR().create(1, "rbf", 0, 2371) #MELHOR H-3 ENERGIA
    # bestSVR_E_SClima = IndividuoSVR().create(1, "rbf", 0, 1917) #MELHOR H-6 ENERGIA
    #
    # bestRF_A_Clima = IndividuoRF().create(3, 6, 143, (12 * 100 * 200)) #MELHOR H-12 AGUA
    # bestRF_A_SClima = IndividuoRF().create(4, 3, 147, (12 * 200 * 500)) #MELHOR H-3 AGUA
    #
    bestRF_E_Clima = IndividuoRF().create(4, 33, 114, (12 * 500 * 1000))  # MELHOR H-12 ENERGIA
    # bestRF_E_SClima = IndividuoRF().create(1, 181, 125, (12 * 200 * 500))

    best = bestRF_E_Clima

    if previsao == "água":
        print("------------------------ INICIANDO TESTE DO CONSUMO DE ÁGUA ------------------------ ")
        df = dfAgua
    elif previsao == "energia":
        print("------------------------ INICIANDO TESTE DO CONSUMO DE ENERGIA ELÉTRICA ------------------------ ")
        df = dfEnergia

    dfHorasAula = agrupamentoMensal(agrupamentoDiarioMedia(dfHorasAula), datas=df["DATA"], strategy="sum")
    dfClima = agrupamentoMensal(dfClima, datas=df["DATA"], strategy="sum")
    df = df.iloc[1:, :]

    dfMerged = df.merge(dfHorasAula, on="DATA", how="inner")

    if clima:
        dfMerged = dfMerged.merge(dfClima, on="DATA", how="inner")

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
                                                  semente=best.semente)

    plotTreinoTeste(dfResultado,
                    title=f"Consumo de {previsao} {h_previsoes} passos à frente - Treino/Teste {algoritmo}")
    plotHistResiduos(dfResultado.tail(h_previsoes),
                     title=f"Histograma de Resíduos da Previsão do Consumo de {previsao} {h_previsoes} passos à frente - Algoritmo {algoritmo}")
    print(dfResumo.head())

    dfPrevisao = prever(modelo, dfMerged, var, h_previsoes, best.n_lags)

    plotPrevisao(dfPrevisao, dfMerged,
                 title=f"PRevisão do Consumo de {previsao} {h_previsoes} passos à frente - Algoritmo {algoritmo}")
