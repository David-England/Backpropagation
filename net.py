import numpy as np
from backpropagation import Layer
from backpropagation.activation_functions import ActivationFunction


class Net:
    def __init__(self, dim_data: int, list_layer_sizes: list[int],
                 list_activations: list[ActivationFunction], d_cost, batch_size=1):
        for x in list_layer_sizes:
            if (type(x) != int):
                raise TypeError("Net: please specify the node layer sizes using only integers.")
            
        if len(list_layer_sizes) != len(list_activations):
            raise ValueError("Net: all supplied lists must be the same length.")
        
        self.d_cost = d_cost
        self.dim_data = dim_data
        self.batch_size = batch_size
        
        self.layers = [Layer(list_layer_sizes[0], dim_data, list_activations[0], batch_size)]
        for i in range(1, len(list_layer_sizes)):
            self.layers.append(Layer(list_layer_sizes[i], list_layer_sizes[i - 1],
                                     list_activations[i], batch_size))
        
    def run(self, x: np.ndarray):
        if x.shape[0] != self.dim_data:
            raise ValueError("Net: data has incorrect number of features.")
        
        a = x
        for i in range(len(self.layers)):
            a = self.layers[i].run(a)

        return a

    def train(self, x: np.ndarray, y_true: np.ndarray):
        if x.shape[0] != self.dim_data:
            raise ValueError("Net: data has incorrect number of features.")
        if y_true.shape[0] != 1:
            raise ValueError("Net: target array should have shape 1 x N.")
        if x.shape[1] != y_true.shape[1]:
            raise ValueError("Net: data and targets have different numbers of data points.")

        nx = x.shape[1]

        for b in range(0, nx - (nx % self.batch_size), self.batch_size):
            self.__train_batch(x[:, b : b + self.batch_size], y_true[:, b : b + self.batch_size])

    def __train_batch(self, x: np.ndarray, y_true: np.ndarray):
        dc_da = self.d_cost(y_true, self.run(x))

        if dc_da.shape != (self.batch_size, self.layers[-1].n_out):
            raise ValueError("Net: d_cost() returns an array of incorrect dimensions.")

        for i in range(len(self.layers), 0, -1):
            dc_da = self.layers[i - 1].calculate_gradient(dc_da)
        
        for lyr in self.layers:
            lyr.update_weights(np.abs(lyr.w).max() / 10.)