import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.graphics.tsaplots as tsaplots


def acf(serie, variaveis):
    for variavel in variaveis:
        tsaplots.plot_acf(serie[variavel], lags=12, title="ACF " + variavel)


def analiseCorrelacao(df):
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
        plt.plot(df["DATA"], df[values], label=values)
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


def plotTreinoTeste(valores, title="", xlabel="Data", ylabel=""):
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(valores["DATA"], valores["PREVISTO"], label='Valores Previstos')
    plt.plot(valores["DATA"], valores["OBSERVADO"], label='Valores Observados')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()
    plt.show()


def plotPrevisao(previsao, original, title="", xlabel="Data", ylabel=""):
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(previsao["DATA"], previsao["PREVISTO"], label='Valores Previstos')
    plt.plot(original["DATA"], original["CONSUMO"], label='Série Original')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

def plotHistResiduos(valores, title="Histograma de Resíduos", xlabel="Resíduos", ylabel="Frequência"):
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(20, 10), dpi=100)
    plt.hist((valores["PREVISTO"] - valores["OBSERVADO"]), bins='auto', color='blue', alpha=0.7)
    plt.gca().set(title=title)
    plt.xlabel('Resíduos')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()