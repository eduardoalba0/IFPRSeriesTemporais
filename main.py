import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


def plotBasico(df, titulo="", xlabel="Data", ylabel=""):
    plt.figure(figsize=(20, 10), dpi=100)
    for values in df.columns.values:
        plt.plot(df.index.values, df[values], label=values)
    plt.gca().set(title=titulo, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()


def plotTreinoTeste(x, valoresPrevistos, valoresEsperados, title="", xlabel="Data", ylabel=""):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(x, valoresPrevistos, label='Valores Previstos')
    plt.plot(x, valoresEsperados, label='Valores Esperados')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()


def prepararDados():
    dfAgua = pd.read_csv("./Dados/ConsumoAguaIFPR.csv", delimiter=";")
    dfEnergia = pd.read_csv("./Dados/ConsumoEnergiaIFPR.csv", delimiter=";")
    dfClima = pd.read_csv("./Dados/ClimaINMETClevelandia.csv", delimiter=";")
    dfAgua["DATA LEITURA"] = pd.to_datetime(dfAgua["DATA LEITURA"], format="%d/%m/%Y")
    dfEnergia["DATA LEITURA"] = pd.to_datetime(dfEnergia["DATA LEITURA"], format="%d/%m/%Y")
    dfClima["DATA"] = pd.to_datetime((dfClima["DATA"] + " " + dfClima["HORA"]),
                                     format="%d/%m/%Y %H:%M")

    dfAgua.set_index("DATA LEITURA", inplace=True)
    dfEnergia.set_index("DATA LEITURA", inplace=True)

    for coluna in dfAgua.columns:
        dfAgua[coluna] = dfAgua[coluna].astype(float)

    for coluna in dfEnergia.columns:
        dfEnergia[coluna] = dfEnergia[coluna].astype(float)

    dfClima.dropna(inplace=True, how="all")
    dfClima.drop("HORA", axis=1, inplace=True)
    for coluna in dfClima.columns[1:]:
        dfClima[coluna] = dfClima[coluna].astype(float)

    return dfAgua, dfEnergia, dfClima


def explorarDados(dfAgua, dfEnergia, dfClima):
    print(dfAgua.columns[dfAgua.isnull().any()])
    print(dfEnergia.columns[dfEnergia.isnull().any()])
    print(dfClima.columns[dfClima.isnull().any()])

    plotBasico(dfAgua,
               titulo="Consumo de Água no IFPR - Campus Palmas no período de Agosto de 2018 a Setembro de 2023")
    plotBasico(dfEnergia,
               titulo="Consumo de Energia Elétrica no IFPR - Campus Palmas no período de Agosto de 2018 a Setembro de 2023")
    plotBasico(dfClima.dropna(),
               titulo="Condições Climáticas da região de Palmas no período de Agosto de 2018 a Setembro de 2023")


def treinarRF(df, var):
    df = df.dropna()
    y = df[var]
    x = df.drop(var, axis=1)

    xTreino, xTeste, yTreino, yTeste = train_test_split(x, y.values, test_size=0.25, random_state=2023)
    modelo = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=2023)
    modelo.fit(xTreino, yTreino)

    print('Conjunto de Treino: %d' % len(yTreino))
    print('Conjunto de Teste: %d' % len(yTeste))

    return modelo, xTreino, xTeste, yTreino, yTeste


def testar(modelo, variaveis, gabarito=None):
    yPrevisto = modelo.predict(variaveis)

    if gabarito is not None and np.any(gabarito):
        mae = mean_absolute_error(gabarito, yPrevisto)
        mape = mean_absolute_percentage_error(gabarito, yPrevisto)
        mse = mean_squared_error(gabarito, yPrevisto)
        rmse = np.sqrt(mse)

        print('MAE: %.3f' % mae)
        print('MAPE: %.3f' % mape)
        print('MSE: %.3f' % mse)
        print('RMSE: %.3f' % rmse)

    return yPrevisto


def preProcessar(df):
    df = df.groupby(pd.Grouper(key="DATA", freq="D")).mean()
    for coluna in df.columns[df.isnull().any()]:
        modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(df, [coluna])
        yPrevisto = testar(modelo, xTeste, yTeste)
        dfAux = pd.DataFrame(xTeste)
        dfAux["PREVISÃO"] = yPrevisto
        dfAux["ESPERADO"] = yTeste

        dfAux.sort_index(inplace=True)
        plotTreinoTeste(dfAux.index.values, dfAux["PREVISÃO"], dfAux["ESPERADO"], title=coluna)


def previsaoAgua(dfAgua):
    modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfAgua, ["CONSUMO (M3)"])
    yPrevisto = testar(modelo, xTeste, yTeste)
    dfAux = pd.DataFrame(xTeste)
    dfAux["PREVISÃO"] = yPrevisto
    dfAux["ESPERADO"] = yTeste

    dfAux.sort_index(inplace=True)
    plotTreinoTeste(dfAux.index.values, dfAux["PREVISÃO"], dfAux["ESPERADO"], title="CONSUMO (M3)")

def previsaoEnergia(dfEnergia):
    modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(dfEnergia, ["ENERGIA ELÉTRICA PONTA (KWh)",
                                                                     "ENERGIA ELÉTRICA FORA DA PONTA (KWh)"])
    yPrevisto = testar(modelo, xTeste, yTeste)
    dfAux = pd.DataFrame(xTeste)

    print(yPrevisto)
    print(yTeste)
    dfPrevisao = dfEnergia.copy()
    dfPrevisao["ENERGIA ELÉTRICA PONTA (KWh)"] = dfPrevisao["ENERGIA ELÉTRICA PONTA (KWh)"].astype(float)
    dfPrevisao["ENERGIA ELÉTRICA FORA DA PONTA (KWh)"] = dfPrevisao["ENERGIA ELÉTRICA FORA DA PONTA (KWh)"].astype(
        float)

    plotTreinoTeste(dfEnergia.index.values,
                    dfPrevisao["ENERGIA ELÉTRICA PONTA (KWh)"],
                    dfEnergia["ENERGIA ELÉTRICA PONTA (KWh)"])

    plotTreinoTeste(dfEnergia.index.values,
                    dfPrevisao["ENERGIA ELÉTRICA FORA DA PONTA (KWh)"],
                    dfEnergia["ENERGIA ELÉTRICA FORA DA PONTA (KWh)"])


dfAgua, dfEnergia, dfClima = prepararDados()
#previsaoEnergia(dfEnergia)
preProcessar(dfClima)
# dfClima = dfClima.groupby(pd.Grouper(key="DATA", freq="M")).mean()
# previsaoEnergia(dfEnergia)
