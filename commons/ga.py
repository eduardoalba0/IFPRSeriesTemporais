import random
import threading


class GA:
    def __init__(self, dados, variaveis, n_individuos, n_geracoes, tx_mutacao, folds=5, semente=None):
        random.seed(semente)
        self.semente = semente
        self.folds = folds
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
        return self.populacao

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
        return individuos

    def crossover(self):
        return self

    def mutation(self, individuo):
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.n_lags = round(individuo.n_lags * (random.uniform(0.5, 1.2)))
            individuo.mutacao = True
        return individuo


class Individuo:
    def __init__(self):
        self.rand_n_lags()
        self.fitness = None
        self.mutacao = False

    def rand_n_lags(self):
        self.n_lags = random.randint(0, 20)
