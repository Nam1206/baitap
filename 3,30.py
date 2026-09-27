import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Lấy dữ liệu Rượu (Wine): Đưa về bài toán nhị phân (Rượu loại 0 -> nhãn 1, Loại khác -> nhãn -1)
X, y = load_wine(return_X_y=True)
y = np.where(y == 0, 1, -1)

# Chia tập train (80%) và test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 2. Lớp Perceptron đơn giản
class Perceptron:

    def __init__(self, eta=0.01, epochs=100):
        self.eta = eta
        self.epochs = epochs

    def fit(self, X, y):
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])
        self.w = np.zeros(X_b.shape[1])

        for _ in range(self.epochs):
            for i in range(len(y)):
                if y[i] * np.dot(self.w, X_b[i]) <= 0:
                    self.w += self.eta * y[i] * X_b[i]

    def predict(self, X):
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.where(np.dot(X_b, self.w) >= 0, 1, -1)


# 3. Huấn luyện và Đánh giá
model = Perceptron()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))