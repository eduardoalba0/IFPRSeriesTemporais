import cudf as pd
import pandas

from commons.preprocessamento import obterLags


def prever(modelo, df, var, h, nLags):
    x = df.copy()
    x["DATA"] = pandas.to_datetime(x["DATA"])
    x["DATA"] = x["DATA"].apply(lambda data: data + pandas.DateOffset(months=h))
    x_previsao = x[["DATA"]].merge(
        x.drop(["DATA"], axis=1).shift(12).tail(h), left_index=True, right_index=True,
        how="inner")

    y_previsto = pandas.DataFrame()
    for index in x_previsao.index:
        if nLags > 0:
            x_teste = obterLags(x, var, nLags).tail(h)
        else:
            x_teste = x_teste.sort_index(axis=1)
        x_previsao = x_previsao.sort_index(axis=1)
        x_aux = pandas.DataFrame(x_teste.loc[index]).transpose()
        y = modelo.predict(x_aux.drop([var, "DATA"], axis=1))
        x.loc[index, var] = int(y)
        y_previsto.loc[index, var] = int(y)

    dfResultado = pandas.DataFrame()

    dfResultado["PREVISTO"] = pandas.concat([df[var], y_previsto[var]], axis=0)
    dfResultado["OBSERVADO"] = df[var]
    dfResultado["DATA"] = pandas.concat([df["DATA"], x_previsao["DATA"]], axis=0)
    return dfResultado
