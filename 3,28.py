import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y = 1

wTx = np.dot(w, x)
y_pred = 1 if wTx > 0 else -1
print('wTx =', wTx)
print('Misclassified:', (y_pred != y))

w_new = w + y * x
print('Updated weights:', w_new)

wTx_new = np.dot(w_new, x)
print('wTx_new =', wTx_new)
y_pred_new = 1 if wTx_new > 0 else -1
print('Misclassified:', (y_pred_new != y))