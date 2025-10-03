import numpy as np

def get_optim_hyperparam(key):
    dictionary = {               ## lr, moment1, moment2
                  'SGD':          [0.01, 0.0, 0.0],  
                  'Momentum_SGD': [0.01, 0.9, 0.0],
                  'NAG':          [0.01, 0.9, 0.0],
                  'Adagrad':      [0.01, 0.0, 0.0],
                  'RMSProp':      [0.001, 0.0, 0.9],
                  'Adam':         [0.001, 0.9, 0.999],
                 }
    return dictionary[key][0], dictionary[key][1], dictionary[key][2]

class Optimizer:
    def __init__(self, otype = 'SGD', lr = 0.001, momentum1 = 0.9, momentum2 = 0.9, eps = 1e-8) -> None:
        self.lr = lr
        self.eps = eps
        self.otype = otype
        self.momentum1 = momentum1
        self.momentum2 = momentum2

    def _ensure_state(self, layer):
        # one velocity per parameter tensor, stored on the layer
        if not hasattr(layer, 'vW'):
            layer.vW = np.zeros_like(layer.weight)
        if not hasattr(layer, 'vb'):
            layer.vb = np.zeros_like(layer.bias)
            
        # for Adagrad & RMSProp & Adam one velocity per parameter tensor, stored on the layer
        if not hasattr(layer, 'gW'):
            layer.gW = np.zeros_like(layer.weight)
        if not hasattr(layer, 'gb'):
            layer.gb = np.zeros_like(layer.bias)
        if not hasattr(layer, 't'):
            layer.t  = 0
        
    def step(self, layer, dW, db):
        self._ensure_state(layer)
        
        if self.otype == 'SGD':
            layer.weight -= self.lr*dW
            layer.bias -= self.lr*db
        elif self.otype == 'Momentum_SGD':
            layer.vW = self.momentum1*layer.vW + dW
            layer.vb = self.momentum1*layer.vb + db
            layer.weight -= self.lr*layer.vW
            layer.bias -= self.lr*layer.vb
        elif self.otype == 'NAG':
            layer.vW = self.momentum1*layer.vW + dW
            layer.vb = self.momentum1*layer.vb + db
            layer.weight -= self.lr*(self.momentum1*layer.vW + dW)
            layer.bias -= self.lr*(self.momentum1*layer.vb + db)
        elif self.otype == 'Adagrad':
            layer.gW += dW**2
            layer.gb += db**2
            layer.weight -= self.lr*dW/(np.sqrt(layer.gW) + self.eps)
            layer.bias -= self.lr*db/(np.sqrt(layer.gb) + self.eps)
        elif self.otype == 'RMSProp':
            layer.gW = self.momentum2*layer.gW + (1 - self.momentum2)*dW**2
            layer.gb = self.momentum2*layer.gb + (1 - self.momentum2)*db**2
            layer.weight -= self.lr*dW/(np.sqrt(layer.gW) + self.eps)
            layer.bias -= self.lr*db/(np.sqrt(layer.gb) + self.eps)
        elif self.otype == 'Adam':
            layer.t += 1
            layer.vW = self.momentum1*layer.vW + (1 - self.momentum1)*dW
            layer.vb = self.momentum1*layer.vb + (1 - self.momentum1)*db
            layer.gW = self.momentum2*layer.gW + (1 - self.momentum2)*dW**2
            layer.gb = self.momentum2*layer.gb + (1 - self.momentum2)*db**2

            vW_hat = layer.vW/(1 - self.momentum1**layer.t)
            vb_hat = layer.vb/(1 - self.momentum1**layer.t)
            gW_hat = layer.gW/(1 - self.momentum2**layer.t)
            gb_hat = layer.gb/(1 - self.momentum2**layer.t)

            layer.weight -= self.lr*vW_hat/(np.sqrt(gW_hat) + self.eps)
            layer.bias -= self.lr*vb_hat/(np.sqrt(gb_hat) + self.eps)
        else:
            raise ValueError("Unknown optimizer")