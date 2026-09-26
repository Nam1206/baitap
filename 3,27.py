import numpy as np

w = np.array([1,2,-10])
x = np.array([3,4,1])
y_true = -1

wTx = np.dot(w, x)
y_pred = 1 if wTx > 0 else -1

is_misclassified = (y_pred != y_true)

print('wTx =', wTx)
print('y_pred =', y_pred)
print('Misclassified:', is_misclassified)