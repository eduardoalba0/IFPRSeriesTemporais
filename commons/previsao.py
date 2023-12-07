import pandas as pd
from dateutil.relativedelta import relativedelta

from commons.preprocessamento import obterLags


def prever(modelo, df, var, h, nLags):
    x = df.copy()
    x = x.reset_index()
    x["DATA"] = pd.to_datetime(x["DATA"])
    x["DATA"] = x["DATA"].apply(lambda data: data + pd.DateOffset(months=h))
    x = x.set_index("DATA")
    x_previsao = x[["ANO", "MES", var]].merge(x.drop(["ANO", "MES", var],axis=1).shift(12).tail((h)), right_index=True, left_index=True, how="inner")

    y_previsto = []
    for index in x_previsao.index:
        if nLags > 0:
            x_previsao = obterLags(x, var, nLags).tail(h)
        else:
            x_previsao = x_previsao.sort_index(axis=1)
        x_previsao = x_previsao.sort_index(axis=1)
        x_aux = pd.DataFrame(x_previsao.loc[index]).transpose()
        y = modelo.predict(x_aux.drop(var, axis=1))
        x.loc[index, "CONSUMO"] = int(y)
        y_previsto.append(y)

    dfResultado = pd.DataFrame()

    dfResultado["PREVISTO"] = pd.concat([df[var], pd.DataFrame(y_previsto, index=x.tail(h).index.values)], axis=0)
    dfResultado["OBSERVADO"] = df[var]
    return dfResultado
