from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def sigma(x):
        pass

    @abstractmethod
    def d_sigma(x):
        pass


class ReLU(ActivationFunction):
    def sigma(x):
        return x if x > 0. else 0.
    
    def d_sigma(x):
        return 1 if x > 0. else 0.
    

class Sigmoid(ActivationFunction):
    def sigma(x):
        return 1. / (1. + np.exp(-x))

    def d_sigma(x):
        return Sigmoid.sigma(x) * (1. - Sigmoid.sigma(x))