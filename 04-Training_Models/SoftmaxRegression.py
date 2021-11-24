import numpy as np

class SoftmaxRegression(object):
    def __init__(self):
        pass

    def fit(self, X, Y, eta = 0.1, epochs = 5000, epsilon = 1e-7):
        np.random.seed(42)
        self.k = Y.max() + 1
        self.m = X.shape[0]
        self.n = X.shape[1] + 1
        # Add bias
        X_b = np.c_[X, np.ones((self.m))]
        Y_hot = self.one_hot(np.array(Y))
        # Init weights
        self.weights = np.random.randn(self.n, self.k)

        for epoch in range(epochs):
            y_hat = X_b.dot(self.weights)
            softmax = self.softmax(y_hat)
            if epoch % 500 == 0:
                loss = -np.mean(np.sum(Y_hot * np.log(softmax + epsilon), axis=1))
                print(epoch, ": ", loss)
            error = softmax - Y_hot
            gradient = 1/self.m * X_b.T.dot(error)
            self.weights = self.weights - eta * gradient

    def predict(self, X):
        X_b = np.c_[X, np.ones((X.shape[0]))]
        y_hat = X_b.dot(self.weights)
        softmax = self.softmax(y_hat)
        return softmax.argmax(axis=1)

    def one_hot(self, x):
        n_cls = x.max() + 1
        leng = x.shape[0]
        output = np.zeros((leng, n_cls))
        output[np.arange(leng), x] = 1
        return output

    def softmax(self, logits):
        # subtracting the max of z for numerical stability.
        exp = np.exp(logits - np.max(logits))
        
        # Calculating softmax for all examples.
        for i in range(len(logits)):
            exp[i] /= np.sum(exp[i])
            
        return exp
