import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from backpropagation import Net
from backpropagation.activation_functions import Sigmoid

# ---- 1. LOADING
x, y = make_classification(n_samples=8192, n_features=4, n_informative=4, n_redundant=0, random_state=4)

# ---- 2. CLEANING
# none required

# ---- 3. TRAIN-TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=4)

# ---- 4. FITTING
n = Net(4, [4, 1], [Sigmoid, Sigmoid], lambda y_true, y_pred: -2. * (y_true - y_pred).T, batch_size=32)
n.train(x_train.T, y_train.reshape(1, -1))

# ---- 5. PREDICTING
filter_point5 = np.vectorize(lambda x: 1 if x > .5 else 0)
predicted = filter_point5(n.run(x_test.T))

print(predicted.reshape(-1)[:25], y_test[:25])