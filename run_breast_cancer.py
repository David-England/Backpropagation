import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from backpropagation import Net
from backpropagation.activation_functions import ReLU, Sigmoid

# ---- 1. LOADING
x, y = load_breast_cancer(return_X_y=True)

# ---- 2. CLEANING
x /= x.mean(axis=0)

# ---- 3. TRAIN-TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=4)

# ---- 4. FITTING
n = Net(30, [30, 4, 1], [Sigmoid, Sigmoid, Sigmoid], lambda y_true, y_pred: -2. * (y_true - y_pred).T, batch_size=4)
n.train(x_train.T, y_train.reshape(1, -1))

# ---- 5. PREDICTING
filter_point5 = np.vectorize(lambda x: 1 if x > .5 else 0)
predicted_unfiltered = n.run(x_test.T)
predicted = filter_point5(predicted_unfiltered)

cm = confusion_matrix(y_test, predicted.reshape(-1))

print(cm)
print(cm.trace() / cm.sum())