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

    # Energia
    individuoRFComClima = IndividuoRF().create(11, 58, 137)
    individuoRFSemClima = IndividuoRF().create(13, 9, 131)
    individuoSVRComClima = IndividuoSVR().create(1, "rbf", 0.1805, 2374)
    individuoSVRSemClima = IndividuoSVR().create(1, "rbf", 0.0477, 2256)

    # Agua
    # individuoRFComClima = IndividuoRF().create(4, 6, 114)
    # individuoRFSemClima = IndividuoRF().create(4, 6, 105)
    # individuoSVRComClima = IndividuoSVR().create(1, "poly", 0.2231, 3148)
    # individuoSVRSemClima = IndividuoSVR().create(0, "poly", 0.3384, 91)

    for dfGlobal, str in [(dfEnergia, "Energia")]:
        df = dfGlobal.copy()
        dfHorasAula = agrupamentoMensal(agrupamentoDiarioMedia(dfHorasAulaGlobal.copy()), datas=df["DATA"],
                                        strategy="sum")
        dfClima = agrupamentoMensal(dfClimaGlobal.copy(), datas=df["DATA"], strategy="sum")
        dfMerged = df.merge(dfHorasAula, on="DATA", how="inner")
        df = df.iloc[1:, :]

        print(f"------------------------ Iniciando teste do consumo de {str} ------------------------ ")
        print("Random Forest")

        for clima, dfMerged, best in [
            ("Com Clima", dfMerged.merge(dfClima, on="DATA", how="inner"), individuoRFComClima),
            ("Sem Clima", dfMerged, individuoRFSemClima)]:
            for h_previsoes in [3, 6, 12]:
                print(f"RF {clima} - {h_previsoes} passos")
                modelo, dfResultado, dfResumo = treinarRF(dfMerged, var, estimators=best.n_estimators,
                                                          maxDepth=best.max_depth, nLags=best.n_lags, folds=h_previsoes)
                plotTreinoTeste(dfResultado,
                                title=f"Consumo de {str} {h_previsoes} passos à frente - Treino/Teste RF {clima}")
                # plotHistResiduos(dfResultado.tail(h_previsoes),
                #                  title=f"Histograma de Resíduos da Previsão do Consumo de {str} {h_previsoes} passos à frente - RF {clima}")
                print(dfResumo.head())

        print(f"------------------------ Iniciando teste do consumo de {str} ------------------------ ")
        print("Support Vector Regression")

        for clima, dfMerged, best in [
            ("Com Clima", dfMerged.merge(dfClima, on="DATA", how="inner"), individuoSVRComClima),
            ("Sem Clima", dfMerged, individuoSVRSemClima)]:
            for h_previsoes in [3, 6, 12]:
                print(f"SVR {clima} - {h_previsoes} passos")
                modelo, dfResultado, dfResumo = treinarSVR(dfMerged, var, kernel=best.kernel,
                                                           epsilon=best.epsilon, c=best.c,
                                                           nLags=best.n_lags, folds=h_previsoes)
                plotTreinoTeste(dfResultado,
                                title=f"Consumo de {str} {h_previsoes} passos à frente - Treino/Teste SVR {clima}")
                # plotHistResiduos(dfResultado.tail(h_previsoes),
                #                  title=f"Histograma de Resíduos da Previsão do Consumo de {str} {h_previsoes} passos à frente - SVR {clima}")
                print(dfResumo.head())
