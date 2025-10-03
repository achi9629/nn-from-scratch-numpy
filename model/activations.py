import numpy as np

class ReLU:
    def __init__(self) -> None:
        pass
    def forward(self, x):
        self.x = x
        x_out = np.maximum(0.0, self.x)
        return x_out
    def backward(self, grad_output):
        grad_input = grad_output * (self.x > 0).astype(np.float32)
        return grad_input

class Softmax:
    def __init__(self) -> None:
        pass
    def forward(self, x):
        x = np.atleast_2d(x)  # ensures 2D
        # subtract max for numerical stability
        shift = x - np.max(x, axis = 1, keepdims = True)
        exps = np.exp(shift)
        return exps/np.sum(exps, axis=1, keepdims = True)