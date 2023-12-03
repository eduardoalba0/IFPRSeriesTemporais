import random
import threading

from commons.ga import GA, Individuo
from commons.treinoTeste import treinarRF


class GARF(GA):
    def init_populacao(self):
        self.populacao = []
        for _ in range(self.n_individuos):
            individuo = IndividuoRF()
            self.populacao.append(individuo)

    def thread_calc_fitness(self, individuos):
        for individuo in individuos:
            modelo, dfResultado, dfResumo = treinarRF(self.dados, self.variaveis, individuo.n_estimators,
                                                                 individuo.max_depth, individuo.min_sample_leaf,
                                                                 individuo.n_lags, self.folds)
            individuo.fitness = dfResumo.loc[0, "MSE"]

    def crossover(self):
        pai = random.choice(self.populacao)
        mae = random.choice(self.populacao)
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


class IndividuoRF(Individuo):
    def __init__(self):
        super().__init__()
        self.rand_n_estimators()
        self.rand_max_depth()
        self.rand_min_sample_leaf()

    def rand_n_estimators(self):
        self.n_estimators = random.randint(2, 200)

    def rand_min_sample_leaf(self):
        self.min_sample_leaf = random.randint(1, 50)

    def rand_max_depth(self):
        self.max_depth = random.randint(50, 200)

    def __str__(self):
        return f'fitness (MAPE): {self.fitness} n_lags: {self.n_lags}, n_estimators: {self.n_estimators}, max_depth: {self.max_depth}, min_sample_leaf: {self.min_sample_leaf}, mutacao: {self.mutacao}'
