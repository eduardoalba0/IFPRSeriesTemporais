import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from commons.treinoTeste import treinarRF, testar


class GARF:
    def __init__(self, dados, variaveis, n_individuos, n_geracoes, semente):
        self.dados = dados
        self.variaveis = variaveis
        self.n_individuos = n_individuos
        self.n_geracoes = n_geracoes
        self.semente = semente
        self.init_populacao()

    def run(self):
        while not self.stop_condition:
            self.fitness_populacao()
            self.best()
            self.crossover()
            self.mutation()

    def init_populacao(self):
        self.populacao = []
        for _ in range(self.n_individuos):
            # Cada lag gerado irá comprometer 1 dado de treinamento, não é bom exagerar
            n_lags = random.randint(0, self.dados.shape[0] / 2)
            # n_estimators é o número de árvores com combinações diferentes de dados e comparações (50 ou menos já geram previsões precisas. mais de 200 aumenta o tempo de execução)
            n_estimators = random.randint(2, 200)
            # max_depth não pode ultrapassar o número de variáveis
            max_depth = random.randint(1, self.dados.shape[1] + n_lags)
            # min_sample_leaf não pode ser maior do que os valores de entrada (observações)
            min_sample_leaf = random.randint(1, self.dados.shape[0])
            individuo = IndividuoRF(n_estimators, max_depth, min_sample_leaf, n_lags)
            self.populacao.append(individuo)

    def fitness_populacao(self):
        for individuo in self.populacao:
            if individuo.fitness == None:
                modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(self.dados, self.variaveis, individuo.n_estimators,
                                                                     individuo.max_depth, individuo.min_sample_leaf,
                                                                     individuo.n_lags)
                dfResultado, dfResumo = testar(modelo, xTeste, yTeste)
                individuo.fitness = dfResumo.loc[0, "MSE"]

    def best(self):
        # Encontra o melhor indivíduo da população
        melhor_individuo = self.populacao[0]
        for individuo in self.populacao:
            if individuo.fitness > melhor_individuo.fitness:
                melhor_individuo = individuo

        return melhor_individuo

    def tournament(self):
        # Seleciona dois indivíduos da população aleatoriamente
        individuo_1 = random.choice(self.populacao)
        individuo_2 = random.choice(self.populacao)

        # Compara os dois indivíduos e retorna o melhor
        if individuo_1.fitness > individuo_2.fitness:
            return individuo_1
        else:
            return individuo_2

    def crossover(self, individuo_1, individuo_2):
        # Seleciona um ponto de corte aleatório
        ponto_de_corte = random.randint(0, len(individuo_1.PARAMETROS) - 1)

        # Cria dois novos indivíduos
        individuo_filho_1 = individuo_1[:ponto_de_corte] + individuo_2[ponto_de_corte:]
        individuo_filho_2 = individuo_2[:ponto_de_corte] + individuo_1[ponto_de_corte:]

        return individuo_filho_1, individuo_filho_2

    def mutation(self, individuo):
        # Seleciona um parâmetro aleatório
        parametro = random.choice(individuo.parametros)

        # Altera o valor do parâmetro
        individuo.parametros[parametro] = random.uniform(0, 1)


class IndividuoRF:

    def __init__(self, n_estimators, max_depth, min_sample_leaf, n_lags):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.n_lags = n_lags
