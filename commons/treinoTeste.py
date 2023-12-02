import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from commons.preprocessamento import obterLags


def treinarRF(df, var, estimators=300, maxDepth=7, minSampleLeaf=1, nLags=0, semente=1234):
    df = obterLags(df, var, lags=nLags)

    df = df.dropna()
    if "DATA" in df.columns:
        df = df.drop("DATA", axis=1)

    y = df[var]
    x = df.drop(var, axis=1)

    xTreino, xTeste, yTreino, yTeste = train_test_split(x, y, test_size=0.2, random_state=semente)
    modelo = RandomForestRegressor(n_estimators=estimators, max_depth=maxDepth, min_samples_leaf=minSampleLeaf,
                                   random_state=2023)
    modelo.fit(xTreino, yTreino)

    return modelo, xTreino, xTeste, yTreino, yTeste


def treinarSVR(df, var, kernel='rbf', C=1.0, epsilon=0.1):
    df = df.dropna()
    if "DATA" in df.columns:
        df = df.drop("DATA", axis=1)
    y = df[var]
    x = df.drop(var, axis=1)
    xTreino, xTeste, yTreino, yTeste = train_test_split(x, y, test_size=0.2, random_state=2023)
    modelo = SVR(kernel=kernel, C=C, epsilon=epsilon)
    modelo.fit(xTreino, yTreino)
    return modelo, xTreino, xTeste, yTreino, yTeste


def testar(modelo, x, yEsperado=None):
    yPrevisto = modelo.predict(x)

    dfResultado = pd.DataFrame(x)
    dfResumo = pd.DataFrame()
    yPrevisto = pd.DataFrame(yPrevisto)
    if yEsperado is not None and np.any(yEsperado):
        dfResultado["ESPERADO"] = yEsperado
        dfResultado["PREVISTO"] = yPrevisto.values
        dfResumo.loc[0, "MAE (KWh)"] = mean_absolute_error(yEsperado, yPrevisto.values)
        dfResumo.loc[0, "MAPE (%)"] = mean_absolute_percentage_error(yEsperado, yPrevisto.values) * 100
        dfResumo.loc[0, "MSE (KWh)"] = mean_squared_error(yEsperado, yPrevisto.values)
        dfResumo.loc[0, "RMSE (KWh)"] = np.sqrt(mean_squared_error(yEsperado, yPrevisto.values))
        dfResumo = dfResumo.round(decimals=2)

    dfResultado = dfResultado.sort_index()
    return dfResultado, dfResumo
