import pandas as pd
from dateutil.relativedelta import relativedelta

from commons.preprocessamento import obterLags


def prever(modelo, df, var, h, lags):
    x = df.copy()
    x = x.reset_index()
    x["DATA"] = x["DATA"].apply(lambda data: data + pd.DateOffset(months=h))
    x["MES"] = x['DATA'].dt.month
    x["ANO"] = x['DATA'].dt.year
    x[var] = x[var].shift(0 - h)
    x = x.set_index("DATA")

    y_previsto = []
    for index in x.tail(h).index:
        if lags > 0:
            for i in range(1, lags + 1):
                lag = x[var].shift(i)
                x[f'{var}_LAG_{i}'] = lag
        x = x.sort_index(axis=1)
        x_previsao = pd.DataFrame(x.loc[index, :].drop(var)).T
        y = modelo.predict(x_previsao)
        x.loc[index, "CONSUMO"] = y
        y_previsto.append(y)

    dfResultado = pd.DataFrame()

    dfResultado["PREVISTO"] = pd.concat([df[var], pd.DataFrame(y_previsto, index=x.tail(h).index.values)], axis=0)
    dfResultado["OBSERVADO"] = df[var]
    return dfResultado
