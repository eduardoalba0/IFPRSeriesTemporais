import seaborn as sns
from matplotlib import pyplot as plt


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
