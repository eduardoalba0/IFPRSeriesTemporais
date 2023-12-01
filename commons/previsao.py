import pandas as pd


def prever(modelo, x, vars):
    x = x.drop(vars, axis=1)
    x = x.dropna()
    yPrevisto = modelo.predict(x)
    yPrevisto = pd.DataFrame(yPrevisto, columns=vars)
    dfResultado = pd.DataFrame(x)
    for column in yPrevisto.columns:
        dfResultado.loc[:, column] = yPrevisto[column].values
    return dfResultado
