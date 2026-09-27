import numpy as np


class Perceptron:

    def __init__(self, eta=0.1, epochs=10):
        self.eta = eta
        self.epochs = epochs
        self.w = None

    def fit(self, X, y):
        # Thêm cột 1 vào X làm bias
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])
        self.w = np.zeros(X_b.shape[1])

        # Huấn luyện mô hình
        for _ in range(self.epochs):
            for i in range(len(y)):
                # Nếu phân lớp sai: y_i * (w^T * x_i) <= 0
                if y[i] * np.dot(self.w, X_b[i]) <= 0:
                    self.w += self.eta * y[i] * X_b[i]

    def predict(self, X):
        # Thêm cột 1 làm bias cho dữ liệu mới
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.where(np.dot(X_b, self.w) >= 0, 1, -1)

if __name__ == "__main__":
    X_train = np.array([[2, 3], [1, 2], [-2, -1], [-3, -2]])
    y_train = np.array([1, 1, -1, -1])

    # 1. Khởi tạo và huấn luyện
    model = Perceptron(eta=1, epochs=10)
    model.fit(X_train, y_train)

    # 2. In trọng số tìm được (w1, w2, bias)
    print("Trọng số w =", model.w)

    # 3. Dự báo dữ liệu mới
    X_test = np.array([[3, 4], [-1, -3]])
    print("Dự báo nhãn =", model.predict(X_test))