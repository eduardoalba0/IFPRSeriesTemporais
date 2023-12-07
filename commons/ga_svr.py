import random
import threading
from commons.treinoTeste import treinarSVR


class GASVR:
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
            print("Geracao: ", _)
        return self.populacao

    def init_populacao(self):
        self.populacao = []
        for _ in range(self.n_individuos):
            individuo = IndividuoSVR()
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
            modelo, dfResultado, dfResumo = treinarSVR(self.dados, self.variaveis, individuo.kernel,
                                                       individuo.epsilon, individuo.c, individuo.n_lags,
                                                       self.folds)
            individuo.fitness = dfResumo.loc[0, "MSE"]

    def crossover(self):
        pai = random.choice(self.populacao)
        mae = random.choice(self.populacao)
        filho = IndividuoSVR()
        filho.n_lags = random.choice([pai.n_lags, mae.n_lags])
        filho.kernel = random.choice([pai.kernel, mae.kernel])
        filho.epsilon = random.choice([pai.epsilon, mae.epsilon])
        filho.c = random.choice([pai.c, mae.c])

        return filho

    def mutation(self, individuo):
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.n_lags = round(individuo.n_lags * (random.uniform(0.5, 1.2)))
            individuo.mutacao = True
            if individuo.n_lags > 24:
                individuo.n_lags = 24
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.n_lags = round(individuo.n_lags * random.uniform(0.5, 1.2))
            individuo.mutacao = True
            if individuo.n_lags > 24:
                individuo.n_lags = 24
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.kernel = random.choice(["poly", "sigmoid", "rbf"])
            individuo.mutacao = True
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.epsilon = individuo.epsilon * random.uniform(0.5, 1.2)
            individuo.mutacao = True
        if random.uniform(0, 1) < self.tx_mutacao:
            individuo.c = round(individuo.c * random.uniform(0.5, 1.2))
            if individuo.c == 0:
                individuo.c = 1
            individuo.mutacao = True
        return individuo


class IndividuoSVR:
    def __init__(self):
        self.rand_n_lags()
        self.rand_kernel()
        self.rand_epsilon()
        self.rand_c()
        self.fitness = None
        self.mutacao = False

    def rand_n_lags(self):
        self.n_lags = random.randint(0, 20)

    def rand_kernel(self):
        self.kernel = random.choice(["poly", "sigmoid", "rbf"])

    def rand_epsilon(self):
        self.epsilon = random.uniform(0, 1)

    def rand_c(self):
        self.c = random.uniform(0, 3000)

    def __str__(self):
        return f'fitness (MAPE): {self.fitness}, n_lags: {self.n_lags}, kernel: {self.kernel}, epsilon: {self.epsilon}, C: {self.c}, mutacao: {self.mutacao}'
