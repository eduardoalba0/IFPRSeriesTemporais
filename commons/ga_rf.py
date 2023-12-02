import random
import threading

from commons.treinoTeste import treinarRF, testar


class GARF:
    def __init__(self, dados, variaveis, n_individuos, n_geracoes, tx_mutacao, semente=None):
        random.seed(semente)
        self.dados = dados
        self.variaveis = variaveis
        self.n_individuos = n_individuos
        self.n_geracoes = n_geracoes
        self.tx_mutacao = tx_mutacao

    def run(self):
        self.init_populacao()
        for _ in range(self.n_geracoes):
            self.calcular_fitness()
            filho = self.crossover()
            filho = self.mutation(filho)
            self.populacao.append(filho)
            self.populacao.pop(len(self.populacao) - 1)
        return self.populacao[0]

    def init_populacao(self):
        self.populacao = []
        for _ in range(self.n_individuos):
            individuo = IndividuoRF()
            self.populacao.append(individuo)

    def calcular_fitness(self):
        print(self.populacao)
        threads = []
        for individuo in filter(lambda individuo: individuo.fitness == None, self.populacao):
            thread = threading.Thread(target=self.thread_calc_fitness(individuo))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.populacao = sorted(self.populacao, key=lambda individuo: individuo.fitness)

    def thread_calc_fitness(self, individuo):
        modelo, xTreino, xTeste, yTreino, yTeste = treinarRF(self.dados, self.variaveis, individuo.n_estimators,
                                                             individuo.max_depth, individuo.min_sample_leaf,
                                                             individuo.n_lags)
        dfResultado, dfResumo = testar(modelo, xTeste, yTeste)
        individuo.fitness = dfResumo.loc[0, "MSE (KWh)"]

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
            individuo.n_lags = round(individuo.n_lags * (random.uniform(0.8, 1.2)))
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.n_estimators = round(individuo.n_estimators * (random.uniform(0.8, 1.2)))
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.max_depth = round(individuo.max_depth * (random.uniform(0.8, 1.2)))
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.min_sample_leaf = round(individuo.min_sample_leaf * (random.uniform(0.8, 1.2)))
        return individuo


class IndividuoRF:
    def __init__(self):
        self.rand_n_lags()
        self.rand_n_estimators()
        self.rand_max_depth()
        self.rand_min_sample_leaf()
        self.fitness = None

    def rand_n_lags(self):
        self.n_lags = random.randint(0, 20)

    def rand_n_estimators(self):
        self.n_estimators = random.randint(2, 200)

    def rand_min_sample_leaf(self):
        self.min_sample_leaf = random.randint(1, 50)

    def rand_max_depth(self):
        self.max_depth = random.randint(0, 200)
        if self.max_depth == 0:
            self.max_depth = None

    def __str__(self):
        return f'n_lags: {self.n_lags}, n_estimators: {self.n_estimators}, max_depth: {self.max_depth}, min_sample_leaf: {self.min_sample_leaf}'
