import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from tabulate import tabulate


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
    dfClima.set_index("DATA", inplace=True)

    return dfAgua, dfEnergia, dfClima


def plotBasico(x, y, titulo="", xlabel="Data", ylabel=""):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(x, y, color="tab:blue")
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


def validar(df, var):
    y = df[var]
    x = df.drop(var, axis=1)
    xTreino, xTeste, yTreino, yTeste = train_test_split(x, y, test_size=0.3, random_state=2023)

    modelo = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=2023)
    modelo.fit(xTreino, yTreino)
    yPrevisao = modelo.predict(xTeste)
    tamTreino = len(xTreino)
    tamTeste = len(xTeste)

    mae = mean_absolute_error(yTeste, yPrevisao)
    mape = mean_absolute_percentage_error(yTeste, yPrevisao)
    mse = mean_squared_error(yTeste, yPrevisao)
    rmse = np.sqrt(mse)

    return mae, mape, mse, rmse, pd.DataFrame(yPrevisao).rename(columns={0: var}), pd.DataFrame(yTeste), tamTreino, tamTeste


dfAgua, dfEnergia, dfClima = prepararDados()

mae, mape, mse, rmse, valoresPrevistos, valoresEsperados, tamTreino, tamTeste = validar(dfEnergia, "CONSUMO (KWh)")
print('Conjunto de Treino: %d' % tamTreino)
print('Conjunto de Teste: %d' % tamTeste)
print('MAE: %.3f' % mae)
print('MAPE: %.3f' % mape)
print('MSE: %.3f' % mse)
print('RMSE: %.3f' % rmse)

valoresPrevistos["DATA LEITURA"] = valoresEsperados.index.values
dfPrevisao = dfEnergia.copy()
for val, data in valoresPrevistos.values:
    dfPrevisao.loc[data, "CONSUMO (KWh)"] = val

plotTreinoTeste(dfEnergia.index.values,
                dfEnergia["CONSUMO (KWh)"],
                dfPrevisao["CONSUMO (KWh)"])

'''
plotBasico(x=dfAgua["DATA LEITURA"], y=dfAgua['CONSUMO (M3)'],
           title="Consumo de Água no IFPR - Campus Palmas no período de Agosto de 2018 a Setembro de 2023")
plotBasico(x=dfEnergia["DATA LEITURA"], y=dfEnergia['CONSUMO (KWh)'],
           title="Consumo de Energia Elétrica no IFPR - Campus Palmas no período de Agosto de 2018 a Setembro de 2023")
plotBasico(x=dfClima["DATA (YYYY-MM-DD)"], y=dfClima["TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)"],
           title="Condições Climáticas da região de Palmas no período de Agosto de 2018 a Setembro de 2023")
'''
