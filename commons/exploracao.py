import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.graphics.tsaplots as tsaplots
from scipy.stats import kruskal
from statsmodels.tsa.stattools import adfuller, kpss

from mktest.mktest import mk_test


def acf(serie, variaveis):
    serie = serie.reset_index()
    for variavel in variaveis:
        tsaplots.plot_acf(serie[variavel], lags=12, title="ACF " + variavel)

def analiseCorrelacao(df):
    df = df.reset_index(drop=True)
    df = df.dropna()

    if "DATA" in df.columns:
        df = df.drop("DATA", axis=1)

    matrizCorrelacao = df.corr().abs()
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrizCorrelacao, cmap='coolwarm')
    plt.show()
    return matrizCorrelacao


def plotBasico(df, titulo="", xlabel="Data", ylabel=""):
    plt.figure(figsize=(20, 10), dpi=100)
    for values in df.columns.values:
        plt.plot(df.index.values, df[values], label=values)
    plt.gca().set(title=titulo, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()


def explorarDados(dfAgua, dfEnergia, dfClima):
    print(dfAgua.columns[dfAgua.isnull().any()])
    print(dfEnergia.columns[dfEnergia.isnull().any()])
    print(dfClima.columns[dfClima.isnull().any()])

    plotBasico(dfAgua,
               titulo="Consumo de Água no IFPR - Campus Palmas no período de Agosto de 2018 a Setembro de 2023")
    plotBasico(dfEnergia,
               titulo="Consumo de Energia Elétrica no IFPR - Campus Palmas no período de Agosto de 2018 a Setembro de 2023")
    plotBasico(dfClima.dropna(),
               titulo="Condições Climáticas da região de Palmas no período de Agosto de 2018 a Setembro de 2023")


def plotTreinoTeste(x, valoresPrevistos, valoresEsperados, title="", xlabel="Data", ylabel=""):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(x, valoresPrevistos, label='Valores Previstos')
    plt.plot(x, valoresEsperados, label='Valores Esperados')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
