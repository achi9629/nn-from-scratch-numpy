import numpy as np

class NLL:
    def __init__(self, eps=1e-12) -> None:
        self.eps = eps
        
    def forward(self, X, Y):
        # X: softmax probabilities, shape [N, C]
        self.X = X
        self.Y = Y
        N = self.X.shape[0]
        p = self.X[np.arange(N), self.Y]
        return -np.log(np.clip(p, self.eps, 1)).mean()
        
    def backward(self):
        # Returns dL/dZ (logits), using X=softmax(Z)
        N = self.X.shape[0]
        grad = self.X.copy()
        grad[np.arange(N), self.Y] -= 1
        grad /= N
        return grad  