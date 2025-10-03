import numpy as np

def weight_init_xavier(fan_in, fan_out, distribution = 'normal'):
    if distribution == 'normal':
        var = np.sqrt(2/(fan_in + fan_out))
        weight = np.random.normal(0, var, (fan_in, fan_out)).astype(np.float32)
    elif distribution == 'uniform':
        a = np.sqrt(6/(fan_in + fan_out))
        weight = np.random.uniform(-a, a, (fan_in, fan_out)).astype(np.float32)
    return weight

class Linear_Layer:
    def __init__(self, input_dim, output_dim) -> None:
        self.weight = weight_init_xavier(input_dim, output_dim)
        self.bias = np.zeros((1, output_dim), dtype = np.float32)
    def forward(self, x):
        self.x = x
        self.out = self.x@self.weight + self.bias
        return self.out
    def backward(self, dz):
        dW = self.x.T@dz
        db = np.sum(dz, 0, keepdims = True)
        dX = dz@self.weight.T
        return dW, db, dX