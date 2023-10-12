import matplotlib.pyplot as plt


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
