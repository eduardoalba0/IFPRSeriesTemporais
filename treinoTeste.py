import matplotlib.pyplot as plt
import numpy as np
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

    xTreino, xTeste, yTreino, yTeste = train_test_split(x, y, test_size=0.3, random_state=2023, shuffle=False)
    modelo = SVR(kernel="linear")
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


def plotTreinoTeste(x, valoresPrevistos, valoresEsperados, title="", xlabel="Data", ylabel=""):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(x, valoresPrevistos, label='Valores Previstos')
    plt.plot(x, valoresEsperados, label='Valores Esperados')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
