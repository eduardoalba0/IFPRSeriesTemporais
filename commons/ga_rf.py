import multiprocessing
import random
import threading

import pandas as pd

from commons.treinoTeste import treinarRF, testar


class GARF:
    def __init__(self, dados, variaveis, n_individuos, n_geracoes, tx_mutacao, semente=None):
        random.seed(semente)
        self.semente= semente
        self.dados = dados
        self.variaveis = variaveis
        self.n_individuos = n_individuos
        self.n_geracoes = n_geracoes
        self.tx_mutacao = tx_mutacao

    def run(self):
        self.init_populacao()
        self.calcular_fitness()
        for _ in range(self.n_geracoes):
            filho = self.crossover()
            filho = self.mutation(filho)
            self.populacao.append(filho)
            self.calcular_fitness()
            self.populacao.pop(len(self.populacao) - 1)
        return self.populacao

    def init_populacao(self):
        self.populacao = []
        for _ in range(self.n_individuos):
            individuo = IndividuoRF()
            self.populacao.append(individuo)

    def calcular_fitness(self):
        populacao_filtrada = list(filter(lambda ind: ind.fitness is None, self.populacao))
        tam_parte = len(populacao_filtrada) // 2
        metade_1 = threading.Thread(target=self.thread_calc_fitness(populacao_filtrada[0:tam_parte]))
        metade_2 = threading.Thread(target=self.thread_calc_fitness(populacao_filtrada[tam_parte:]))
        metade_1.start()
        metade_2.start()
        metade_1.join()
        metade_2.join()

        self.populacao = sorted(self.populacao, key=lambda ind: ind.fitness)

    def thread_calc_fitness(self, individuos):
        for individuo in individuos:
            modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(self.dados, self.variaveis, individuo.n_estimators,
                                                                 individuo.max_depth, individuo.min_sample_leaf,
                                                                 individuo.n_lags, self.semente)
            dfResultado, dfResumo = testar(modelo, xTeste, yTeste)
            individuo.fitness = dfResumo.loc[0, "MSE (KWh)"]

    def crossover(self):
        pai = self.populacao[0]
        mae = self.populacao[1]
        filho = IndividuoRF()
        filho.n_lags = random.choice([pai.n_lags, mae.n_lags])
        filho.n_estimators = random.choice([pai.n_estimators, mae.n_estimators])
        filho.max_depth = random.choice([pai.max_depth, mae.max_depth])
        filho.min_sample_leaf = random.choice([pai.min_sample_leaf, mae.min_sample_leaf])

        return filho

    def mutation(self, individuo):
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.n_lags = round(individuo.n_lags * (random.uniform(0.5, 1.2)))
            individuo.mutacao =  True
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.n_estimators = round(individuo.n_estimators * (random.uniform(0.5, 1.2)))
            individuo.mutacao = True
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.max_depth = round(individuo.max_depth * (random.uniform(0.5, 1.2)))
            individuo.mutacao = True
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.min_sample_leaf = round(individuo.min_sample_leaf * (random.uniform(0.5, 1.2)))
            individuo.mutacao = True
        return individuo


class IndividuoRF:
    def __init__(self):
        self.rand_n_lags()
        self.rand_n_estimators()
        self.rand_max_depth()
        self.rand_min_sample_leaf()
        self.fitness = None
        self.mutacao = False

    def rand_n_lags(self):
        self.n_lags = random.randint(0, 20)

    def rand_n_estimators(self):
        self.n_estimators = random.randint(2, 200)

    def rand_min_sample_leaf(self):
        self.min_sample_leaf = random.randint(1, 50)

    def rand_max_depth(self):
        self.max_depth = random.randint(50, 200)

    def __str__(self):
        return f'fitness (MAPE): {self.fitness} n_lags: {self.n_lags}, n_estimators: {self.n_estimators}, max_depth: {self.max_depth}, min_sample_leaf: {self.min_sample_leaf}, mutacao: {self.mutacao}'
