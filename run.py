import numpy as np
from sklearn.datasets import make_classification
from backpropagation import Net, ReLU

x, y = make_classification(n_samples=100, n_features=4, n_informative=4, n_redundant=0, random_state=4)

n = Net(4, [4, 1], [ReLU, ReLU], lambda y_true, y_pred: -2. * (y_true - y_pred).T, batch_size=2)

print("\nWeights BEFORE")
print(n.layers[0].w)
print(n.layers[1].w)
print(f"Max weights: {np.abs(n.layers[0].w).max()}, {np.abs(n.layers[1].w).max()}")

n.train(x[:2, :].T, y.reshape(1, 100)[:, :2])

print("\nInput/output values")
print(n.layers[0].a_in)
print(n.layers[1].a_in)

print("\nLinear results (z)")
print(n.layers[0].z)
print(n.layers[1].z)

print("\nWeights")
print(n.layers[0].w)
print(n.layers[1].w)
print(f"Max weights: {np.abs(n.layers[0].w).max()}, {np.abs(n.layers[1].w).max()}")

print("\nWeight gradients")
print(n.layers[0].dc_dw)
print(n.layers[1].dc_dw)

print(f"\ny (actual): {y[0]}")