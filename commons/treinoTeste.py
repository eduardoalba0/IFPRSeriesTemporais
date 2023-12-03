import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.svm import SVR

from commons.preprocessamento import obterLags


def treinarRF(df, var, estimators, maxDepth, minSampleLeaf, nLags, folds):
    if nLags > 0:
        df = obterLags(df, var, lags=nLags)
    x_testes = pd.DataFrame()
    y_testes = pd.DataFrame()
    y_previstos = pd.DataFrame()

    df = df.dropna()
    if "DATA" in df.columns:
        df = df.drop("DATA", axis=1)

    y = df[var]
    x = df.drop(var, axis=1)

    modelo = RandomForestRegressor(n_estimators=estimators, max_depth=maxDepth, min_samples_leaf=minSampleLeaf,
                                   random_state=2023)

    for treino_index, teste_index in TimeSeriesSplit(n_splits=folds, test_size=1).split(x, y):
        x_treino, x_teste = x.iloc[treino_index], x.iloc[teste_index]
        y_treino, y_teste = y.iloc[treino_index], y.iloc[teste_index]
        modelo.fit(x_treino, y_treino)
        y_previsto = modelo.predict(x_teste)
        x_testes = pd.concat([x_testes, pd.DataFrame(x_teste)], axis=0)
        y_testes = pd.concat([y_testes, pd.DataFrame(y_teste)], axis=0)
        y_previstos = pd.concat([y_previstos, pd.DataFrame(y_previsto)], axis=0)

    dfResultado = pd.DataFrame(x_testes)
    dfResumo = pd.DataFrame()
    yPrevisto = pd.DataFrame(y_previstos)
    if y_testes is not None and np.any(y_testes):
        dfResultado["ESPERADO"] = y_testes
        dfResultado["PREVISTO"] = yPrevisto.values
        dfResumo.loc[0, "MAE"] = mean_absolute_error(y_testes, yPrevisto.values)
        dfResumo.loc[0, "MAPE (%)"] = mean_absolute_percentage_error(y_testes, yPrevisto.values) * 100
        dfResumo.loc[0, "MSE"] = mean_squared_error(y_testes, yPrevisto.values)
        dfResumo.loc[0, "RMSE"] = np.sqrt(mean_squared_error(y_testes, yPrevisto.values))
        dfResumo = dfResumo.round(decimals=2)

    dfResultado = dfResultado.sort_index()

    return modelo, dfResultado, dfResumo


def treinarSVR(df, var, kernel, epsilon, gamma, c, nLags, folds):
    if nLags > 0:
        df = obterLags(df, var, lags=nLags)
    x_testes = pd.DataFrame()
    y_testes = pd.DataFrame()
    y_previstos = pd.DataFrame()

    df = df.dropna()
    if "DATA" in df.columns:
        df = df.drop("DATA", axis=1)

    y = df[var]
    x = df.drop(var, axis=1)

    modelo = SVR(kernel=kernel, epsilon=epsilon, gamma=gamma, C=c)
    for treino_index, teste_index in TimeSeriesSplit(n_splits=folds, test_size=1).split(x, y):
        x_treino, x_teste = x.iloc[treino_index], x.iloc[teste_index]
        y_treino, y_teste = y.iloc[treino_index], y.iloc[teste_index]
        modelo.fit(x_treino, y_treino)
        y_previsto = modelo.predict(x_teste)
        x_testes = pd.concat([x_testes, pd.DataFrame(x_teste)], axis=0)
        y_testes = pd.concat([y_testes, pd.DataFrame(y_teste)], axis=0)
        y_previstos = pd.concat([y_previstos, pd.DataFrame(y_previsto)], axis=0)

    dfResultado = pd.DataFrame(x_testes)
    dfResumo = pd.DataFrame()
    yPrevisto = pd.DataFrame(y_previstos)
    if y_testes is not None and np.any(y_testes):
        dfResultado["ESPERADO"] = y_testes
        dfResultado["PREVISTO"] = yPrevisto.values
        dfResumo.loc[0, "MAE"] = mean_absolute_error(y_testes, yPrevisto.values)
        dfResumo.loc[0, "MAPE (%)"] = mean_absolute_percentage_error(y_testes, yPrevisto.values) * 100
        dfResumo.loc[0, "MSE"] = mean_squared_error(y_testes, yPrevisto.values)
        dfResumo.loc[0, "RMSE"] = np.sqrt(mean_squared_error(y_testes, yPrevisto.values))
        dfResumo = dfResumo.round(decimals=2)

    dfResultado = dfResultado.sort_index()

    return modelo, dfResultado, dfResumo
