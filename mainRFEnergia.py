import time

import numpy as np
import pandas as pd

from commons.exploracao import plotTreinoTeste, plotPrevisao, plotHistResiduos
from commons.ga_rf import GARF, IndividuoRF
from commons.ga_svr import GASVR, IndividuoSVR
from commons.preprocessamento import prepararDados, agrupamentoMensal, agrupamentoDiarioMedia
from commons.previsao import prever
from commons.treinoTeste import treinarSVR, treinarRF

if __name__ == '__main__':
    dfAgua, dfEnergia, dfClimaGlobal, dfHorasAulaGlobal = prepararDados()

    var = "CONSUMO"
    sementes = [123, 1234, 1235, 12346]

    for dfGlobal, str in [(dfEnergia, "Energia Elétrica")]:
        df = dfGlobal.copy()
        dfHorasAula = agrupamentoMensal(agrupamentoDiarioMedia(dfHorasAulaGlobal.copy()), datas=df["DATA"],
                                        strategy="sum")
        dfClima = agrupamentoMensal(dfClimaGlobal.copy(), datas=df["DATA"], strategy="sum")
        dfMerged = df.merge(dfHorasAula, on="DATA", how="inner")
        df = df.iloc[1:, :]

        print(f"------------------------ Iniciando teste do consumo de {str} ------------------------ ")
        print("Random Forest")

        global_populacao = pd.DataFrame()
        global_treino_teste = pd.DataFrame()
        global_metricas = pd.DataFrame()
        global_treino_teste["DATA"] = dfMerged["DATA"]
        global_metricas["MEDIDAS"] = ["MAE", "MAPE (%)", "MSE", "RMSE"]
        global_metricas = global_metricas.set_index("MEDIDAS")
        for clima, dfMerged in [("Com Clima", dfMerged.merge(dfClima, on="DATA", how="inner")), ("Sem Clima", dfMerged)]:
            bests = []
            for individuos, geracoes in [(100, 200), (200, 500), (500, 1000)]:
                populacao = []
                for semente in sementes:
                    print(f"População com {individuos} individuos e {geracoes} gerações.")
                    t_inicio = time.time()
                    ga = GARF(dfMerged, var, individuos, geracoes, 0.5, 12, semente)
                    best = ga.run()[0]
                    t_fim = time.time()
                    best.tempo_execucao = (t_fim - t_inicio)
                    populacao.append(best)
                    print(f"Tempo de execução: {t_fim - t_inicio} segundos")
                populacao = sorted(populacao, key=lambda ind: ind.fitness)
                bests.append(populacao[0])
                global_populacao[f"{str} {individuos}Ind. {geracoes}Ger. - {clima}"] = populacao
                global_populacao.to_csv(f"Resultados/GA-RF {str}.csv", index=False, sep=";")

            bests = sorted(bests, key=lambda ind: ind.fitness)
            best = bests[0]
            for h_previsoes in [3, 6, 12]:
                modelo, dfResultado, dfResumo = treinarRF(dfMerged, var, estimators=best.n_estimators,
                                                          maxDepth=best.max_depth, nLags=best.n_lags, folds=h_previsoes)
                global_treino_teste[f"{str} {h_previsoes} passos - {clima}"] = dfResultado["PREVISTO"]
                global_metricas[f"{h_previsoes} passos - {clima}"] = dfResumo.loc[0]

                global_treino_teste.to_csv(f"Resultados/RF-Treino {str}.csv", index=False, sep=";")
                global_metricas.to_csv(f"Resultados/RF-Métricas {str}.csv", index=False, sep=";")

                # plotTreinoTeste(dfResultado,
                #                 title=f"Consumo de {str} {h_previsoes} passos à frente - Treino/Teste RF")
                # plotHistResiduos(dfResultado.tail(h_previsoes),
                #                  title=f"Histograma de Resíduos da Previsão do Consumo de {str} {h_previsoes} passos à frente - RF")
    # print(dfResumo.head())
    #
    # dfPrevisao = prever(modelo, dfMerged, var, h_previsoes, best.n_lags)
    #
    # plotPrevisao(dfPrevisao, dfMerged,
    #              title=f"Previsão do Consumo de {str} {h_previsoes} passos à frente - RF")
