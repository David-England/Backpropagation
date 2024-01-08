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
        
        self.z = np.dot(self.w, input)
        self.a = np.apply_along_axis(self.sigma, 0, self.z)