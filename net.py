import numpy as np
from backpropagation import Layer


class Net:
    def __init__(self, n_data: int, list_layer_sizes: list[int], list_sigmas, list_d_sigmas,
                 d_cost, batch_size=1):
        for x in list_layer_sizes:
            if (type(x) != int):
                raise TypeError("Net: please specify the node layer sizes using only integers.")
            
        if ((len(list_layer_sizes) != len(list_sigmas))
            or (len(list_layer_sizes) != len(list_d_sigmas))):
            raise ValueError("Net: all supplied lists must be the same length.")
        
        self.d_cost = d_cost
        self.n_data = n_data
        self.batch_size = batch_size
        
        self.layers = [Layer(list_layer_sizes[0], n_data, list_sigmas[0], list_d_sigmas[0],
                             batch_size)]
        for i in range(1, len(list_layer_sizes)):
            self.layers.append(Layer(list_layer_sizes[i], list_layer_sizes[i - 1],
                                     list_sigmas[i], list_d_sigmas[i], batch_size))
        
    def run(self, x: np.ndarray):
        if x.shape != (self.n_data, self.batch_size):
            raise ValueError("Net: data has incorrect dimensions.")
        
        a = x
        for i in range(len(self.layers)):
            a = self.layers[i].run(a)

        return a
        
    def train(self, x: np.ndarray, y_true: np.ndarray):
        dc_da = self.d_cost(y_true, self.run(x))

        if dc_da.shape != (self.batch_size, self.layers[-1].n_out):
            raise ValueError("Net: d_cost() returns an array of incorrect dimensions.")

        for i in range(len(self.layers), 0, -1):
            dc_da = self.layers[i - 1].calculate_gradient(dc_da)
        
        for lyr in self.layers:
            lyr.update_weights(np.abs(lyr.w).max() / 10.)