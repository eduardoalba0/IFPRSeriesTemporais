import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def treinarRF(df, var):
    df = df.dropna()
    if "DATA" in df.columns:
        df = df.drop("DATA", axis=1)

    y = df[var]
    x = df.drop(var, axis=1)

    xTreino, xTeste, yTreino, yTeste = train_test_split(x, y, test_size=0.2, random_state=2023)
    modelo = RandomForestRegressor(n_estimators=300, max_depth=7, random_state=2023)
    modelo.fit(xTreino, yTreino)

    return modelo, xTreino, xTeste, yTreino, yTeste


def testar(modelo, x, yEsperado=None):
    yPrevisto = modelo.predict(x)

    if yEsperado is not None and np.any(yEsperado):
        mae = mean_absolute_error(yEsperado, yPrevisto)
        mape = mean_absolute_percentage_error(yEsperado, yPrevisto)
        mse = mean_squared_error(yEsperado, yPrevisto)
        rmse = np.sqrt(mse)

        print('MAE: %.3f' % mae)
        print(('MAPE: %.3f' % (mape * 100)) + " %")
        print('MSE: %.3f' % mse)
        print('RMSE: %.3f' % rmse)

    return x, yEsperado, yPrevisto

def plotTreinoTeste(x, valoresPrevistos, valoresEsperados, title="", xlabel="Data", ylabel=""):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(x, valoresPrevistos, label='Valores Previstos')
    plt.plot(x, valoresEsperados, label='Valores Esperados')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
