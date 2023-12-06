import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.svm import SVR

from commons.preprocessamento import obterLags


def treinarRF(df, var, estimators, maxDepth, nLags, folds, semente):
    if nLags > 0:
        df = obterLags(df, var, lags=nLags)

    if "DATA" in df.columns:
        df = df.drop("DATA", axis=1)

    df = df.dropna()
    df = df.sort_index(axis=1)

    modelo = RandomForestRegressor(n_estimators=estimators, max_depth=maxDepth,
                                   random_state=semente)

    return treino_teste_validacao_cruzada(df, var, modelo, folds)


def treinarSVR(df, var, kernel, epsilon, c, nLags, folds):
    df = df.drop(columns=df.filter(like='_LAG_').columns, axis=1)
    if nLags > 0:
        df = obterLags(df, var, lags=nLags)
    if "DATA" in df.columns:
        df = df.drop("DATA", axis=1)

    df = df.dropna()

    modelo = SVR(kernel=kernel, epsilon=epsilon, C=c)

    return treino_teste_validacao_cruzada(df, var, modelo, folds)


def treino_teste_validacao_cruzada(df, var, modelo, folds):
    y = df[var]
    x = df.drop(var, axis=1)
    x_testes = pd.DataFrame()
    y_testes = pd.DataFrame()
    y_previstos = pd.DataFrame()

    for treino_index, teste_index in TimeSeriesSplit(n_splits=max(folds,5), test_size=1).split(x, y):
        x_treino, x_teste = x.iloc[treino_index], x.iloc[teste_index]
        y_treino, y_teste = y.iloc[treino_index], y.iloc[teste_index]
        modelo.fit(x_treino, y_treino)
        y_previsto = modelo.predict(x_teste)
        x_testes = pd.concat([x_testes, pd.DataFrame(x_teste)], axis=0)
        y_testes = pd.concat([y_testes, pd.DataFrame(y_teste)], axis=0)
        y_previstos = pd.concat([y_previstos, pd.DataFrame(y_previsto)], axis=0)

    dfResultado, dfResumo = medidas_desempenho(x_testes, y_testes, y_previstos)

    df["PREVISTO"] = df[var]
    df["OBSERVADO"] = df[var]
    df.update(dfResultado)
    dfResultado = df[["PREVISTO", "OBSERVADO"]]

    return modelo, dfResultado, dfResumo


def treino_teste_sequencial(df, var, modelo, folds):
    y = df[var]
    x = df.drop(var, axis=1)

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=folds, shuffle=False)

    modelo.fit(x_treino, y_treino)
    y_previstos = modelo.predict(x_teste)

    dfResultado, dfResumo = medidas_desempenho(x_teste, y_teste, y_previstos)
    df["PREVISTO"] = df[var]
    df["OBSERVADO"] = df[var]
    df.update(dfResultado)
    dfResultado = df[["PREVISTO", "OBSERVADO"]]
    return modelo, dfResultado, dfResumo


def medidas_desempenho(x_teste, y_teste, y_previsto):
    dfResultado = pd.DataFrame(x_teste)
    dfResumo = pd.DataFrame()
    yPrevisto = pd.DataFrame(y_previsto)
    if y_teste is not None and np.any(y_teste):
        dfResultado["OBSERVADO"] = y_teste
        dfResultado["PREVISTO"] = yPrevisto.values
        dfResumo.loc[0, "MAE"] = mean_absolute_error(y_teste, yPrevisto.values)
        dfResumo.loc[0, "MAPE (%)"] = mean_absolute_percentage_error(y_teste, yPrevisto.values) * 100
        dfResumo.loc[0, "MSE"] = mean_squared_error(y_teste, yPrevisto.values)
        dfResumo.loc[0, "RMSE"] = np.sqrt(mean_squared_error(y_teste, yPrevisto.values))
        dfResumo = dfResumo.round(decimals=2)

    return dfResultado, dfResumo
