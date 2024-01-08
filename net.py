import numpy as np
from backpropagation import Layer


class Net:
    def __init__(self, n_data: int, list_layer_sizes: list[int], list_sigmas, list_d_sigmas, cost):
        for x in list_layer_sizes:
            if (type(x) != int):
                raise TypeError("Net: please specify the node layer sizes using only integers.")
            
        if ((len(list_layer_sizes) != len(list_sigmas))
            or (len(list_layer_sizes) != len(list_d_sigmas))):
            raise ValueError("Net: all supplied lists must be the same length.")
        
        self.cost = cost
        self.n_data = n_data
        
        self.layers = [Layer(list_layer_sizes[0], n_data, list_sigmas[0], list_d_sigmas[0])]
        for i in range(1, len(list_layer_sizes)):
            self.layers.append(Layer(list_layer_sizes[i], list_layer_sizes[i - 1],
                                     list_sigmas[i], list_d_sigmas[i]))