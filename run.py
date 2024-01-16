import numpy as np
from sklearn.datasets import make_classification
from backpropagation import Net

x, y = make_classification(n_samples=100, n_features=4, n_informative=4, n_redundant=0, random_state=4)

sigma = lambda x: x if x > 0 else 0
d_sigma = lambda y: 1 if y > 0 else 0
d_cost = lambda y_true, y_pred: np.array([-2. * (y_true[0, 0] - y_pred[0, 0])]).reshape(1, 1)

n = Net(4, [4, 1], [sigma, sigma], [d_sigma, d_sigma], d_cost)

print("\nWeights BEFORE")
print(n.layers[0].w)
print(n.layers[1].w)
print(f"Max weights: {np.abs(n.layers[0].w).max()}, {np.abs(n.layers[1].w).max()}")

n.train(x[0:1, :].T, y.reshape(100, 1)[0:1, :])

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