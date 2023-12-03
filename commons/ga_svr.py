import random
import threading

from commons.ga import GA, Individuo
from commons.treinoTeste import treinarRF, treinarSVR


class GASVR(GA):
    def init_populacao(self):
        self.populacao = []
        for _ in range(self.n_individuos):
            individuo = IndividuoSVR()
            self.populacao.append(individuo)

    def thread_calc_fitness(self, individuos):
        for individuo in individuos:
            modelo, dfResultado, dfResumo = treinarSVR(self.dados, self.variaveis, individuo.kernel,
                                                       individuo.epsilon, individuo.gamma, individuo.c,
                                                       individuo.n_lags, self.folds)
            individuo.fitness = dfResumo.loc[0, "MSE"]

    def crossover(self):
        pai = random.choice(self.populacao)
        mae = random.choice(self.populacao)
        filho = IndividuoSVR()
        filho.n_lags = random.choice([pai.n_lags, mae.n_lags])
        filho.kernel = random.choice([pai.kernel, mae.kernel])
        filho.epsilon = random.choice([pai.epsilon, mae.epsilon])
        filho.gamma = random.choice([pai.gamma, mae.gamma])
        filho.c = random.choice([pai.c, mae.c])

        return filho

    def mutation(self, individuo):
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.n_lags = round(individuo.n_lags * random.uniform(0.5, 1.2))
            individuo.mutacao = True
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.kernel = random.choice(["poly", "sigmoid", "rbf"])
            individuo.mutacao = True
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.epsilon = individuo.epsilon * random.uniform(0.5, 1.2)
            individuo.mutacao = True
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.c = round(individuo.c * random.uniform(0.5, 1.2))
            individuo.mutacao = True
        return individuo


class IndividuoSVR(Individuo):
    def __init__(self):
        super().__init__()
        self.rand_kernel()
        self.rand_epsilon()
        self.rand_gamma()
        self.rand_c()

    def rand_kernel(self):
        self.kernel = random.choice(["poly", "sigmoid", "rbf"])

    def rand_epsilon(self):
        self.epsilon = random.uniform(0, 1)

    def rand_gamma(self):
        self.gamma = random.choice(["auto", "scale"])

    def rand_c(self):
        self.c = random.uniform(0, 2000)

    def __str__(self):
        return f'fitness (MAPE): {self.fitness}, n_lags: {self.n_lags}, kernel: {self.kernel}, epsilon: {self.epsilon}, gamma: {self.gamma}, C: {self.c}, mutacao: {self.mutacao}'
