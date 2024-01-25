# Working neural nets featuring backpropagation

This is a library for constructing simple neural net models. "Learning" takes place using the "backpropagation" mechanism, ultimately a weaponisation of the differentiation chain rule learnt by A-Level mathematics students: see https://en.wikipedia.org/wiki/Backpropagation. In the library, a *Layer* object represents a layer of nodes in the model and wraps a matrix of weights, among other features. A *Net* contains a list of *Layer*s and is the API the developer is generally expected to use; instantiate it by passing the following elements to the constructor.
* *dim_data*, int: the number of features in the data.
* *list_layer_sizes*, list[int]: specify how many nodes each layer should have by supplying a list of int values.
* *list_activations*, list[ActivationFunction]: the activation function for each node can be specified by passing in a list of elements from the *backpropagation.activation_functions* module.
* *d_cost*, function: any function that computes the derivative of the cost function. The library calls *d_cost*() with two arguments: (*y* (the 1 x *batch_size* target array), output of *Net.run*() (the $n_{lastlayer}$ x *batch_size* output array)).
* *batch_size*, int, default 1: the number of data points to include per batch while training the model. Please note that any "remainder" data points after batching will not be used.

In any case, it is assumed that the input data is a NumPy array of dimension (*dim_data* x *N*), where *N* = *batch_size* when training, and similarly that the targets have dimension (*1* x *batch_size*).

## To do
* Layer weights initialisation: make the initialisation of the weights more intelligent. Perhaps read https://arxiv.org/pdf/1704.08863.pdf.
* Include more activation functions. Currently, only *Sigmoid* and *ReLU* are supported; at the very least, *tanh* should be added.