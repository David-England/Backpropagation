import numpy as np
from backpropagation import ActivationFunction


class Layer:
    def __init__(self, n: int, n_prev: int, activation: ActivationFunction, batch_size=1):
        self.n_in = n_prev
        self.n_out = n
        self.batch_size = batch_size

        self.activation = activation

        np.random.seed(4)
        self.w = np.random.randn(n, n_prev + 1)
    
    def run(self, input: np.ndarray):
        if input.shape[0] != self.n_in:
            raise ValueError("Layer received array of incorrect number of rows.")
        
        self.a_in = input
        self.z = np.dot(self.w, self.__add_intercept_row(input))

        return np.vectorize(self.activation.sigma)(self.z)

    def calculate_gradient(self, dc_da: np.ndarray):
        if dc_da.shape != (self.batch_size, self.n_out):
            raise ValueError("Layer received array of incorrect size during backpropagation.")
        
        dc_dz = dc_da * np.vectorize(self.activation.d_sigma)(self.z.T)
        self.dc_dw = np.dot(dc_dz.T, self.__add_intercept_row(self.a_in).T)

        # Return outgoing dC/da value:
        return np.dot(dc_dz, self.w[:, 1:])
    
    def update_weights(self, step_size: float):
        self.w -= step_size * self.dc_dw
    
    def __add_intercept_row(self, x: np.ndarray):
        return np.concatenate((np.ones((1, x.shape[1])), x))