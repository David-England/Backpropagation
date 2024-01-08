import numpy as np


class Layer:
    def __init__(self, n: int, n_prev: int, sigma, d_sigma):
        self.n_in = n_prev
        self.n_out = n

        self.sigma = sigma
        self.d_sigma = d_sigma

        np.random.seed(4)
        self.w = np.random.randn(n, n_prev)
    
    def run(self, input: np.ndarray):
        if input.shape != (self.n_in, 1):
            raise ValueError("Layer received array of incorrect size.")
        
        self.a_in = input
        self.z = np.dot(self.w, input)

        return np.vectorize(self.sigma)(self.z)

    def calculate_gradient(self, dc_da: np.ndarray):
        if dc_da.shape != (1, self.n_out):
            raise ValueError("Layer received array of incorrect size during backpropagation.")
        
        dc_dz = dc_da * np.vectorize(self.d_sigma)(self.z.T)
        self.dc_dw = np.dot(dc_dz.T, self.a_in.T)

        # Return outgoing dC/da value:
        return np.dot(dc_dz, self.w)
    
    def update_weights(self, step_size: float):
        self.w -= step_size * self.dc_dw