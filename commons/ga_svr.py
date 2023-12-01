import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
class GASVR:
    def __init__(self, dados, n_individuos, n_geracoes, semente):
        self.dados = dados
        self.n_individuos = n_individuos
        self.n_geracoes = n_geracoes
        self.semente = semente
        self.populacao = []

class IndividuoSVR:
    def __init__(self,  n_lags):
        self.n_lags = n_lags