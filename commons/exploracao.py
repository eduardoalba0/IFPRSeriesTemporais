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


def plotTreinoTeste(valoresPrevistos, valoresObservados, title="", xlabel="Data", ylabel=""):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(valoresObservados.index.values, valoresPrevistos, label='Valores Previstos')
    plt.plot(valoresObservados.index.values, valoresObservados, label='Valores Observados')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()
    plt.show()


def plotPrevisao(previsao, original, title="", xlabel="Data", ylabel=""):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(previsao.index.values, previsao, label='Valores Previstos')
    plt.plot(original.index.values, original, label='Série Original')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

def plotHistResiduos(valoresPrevistos, valoresObservados, title="Histograma de Resíduos", xlabel="Resíduos", ylabel="Frequência"):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.hist((valoresObservados - valoresPrevistos), bins='auto', color='blue', alpha=0.7)
    plt.xlabel('Resíduos')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()