import numpy as np


class Perceptron:

    def __init__(self, input_dim=2):
        self.w = np.zeros((input_dim))

    def train(self, x_train, y_train):
        while(True):
            m = 0
            for xi, yi in zip(x_train, y_train):
                res = np.dot(self.w.T, xi)
                if yi * res <= 0:
                    self.w = self.w + yi * xi
                    m += 1
            if m == 0:
                break          

    def predict(self, x):
        return 1 if np.dot(self.w.T, x) > 0 else -1

    def save_weights(self, filename):
        if filename is None:
            return
        with open(filename, 'wb') as file:
            np.save(file, self.w)

    def load_weights(self, filename):
        if filename is None:
            return
        with open(filename, 'rb') as file:
            self.w = np.load(file)

if __name__ == "__main__":
    x = np.array([[-4, 2], [-1, -1], [-2, 4], [2, -1], [-3, 6], [3, 2], [1, 5], [6, 2], [3, 7], [6, 4]])
    y = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])

    p = Perceptron()
    p.train(x, y)
    p.save_weights('weights')
    # p.load_weights('weights')
    print(p.predict([1, -8]))
