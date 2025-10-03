import numpy as np
from model.model_utils import Linear_Layer
from model.activations import ReLU, Softmax

class MLP:
    def __init__(self, layer_dimensions, optim) -> None:

        self.hidden_layer1 = Linear_Layer(layer_dimensions[0], layer_dimensions[1])
        self.hidden_layer2 = Linear_Layer(layer_dimensions[1], layer_dimensions[2])
        self.final_layer = Linear_Layer(layer_dimensions[2], layer_dimensions[3])

        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.softmax = Softmax()
        self.optim = optim

    def step(self):
        self.optim.step(self.final_layer, self.dW3, self.db3)
        self.optim.step(self.hidden_layer2, self.dW2, self.db2)
        self.optim.step(self.hidden_layer1, self.dW1, self.db1)
        
    def forward(self, x):
        self.x = x.reshape(-1, 28*28).astype(np.float32)
        self.x_out_1 = self.relu1.forward(self.hidden_layer1.forward(self.x))
        self.x_out_2 = self.relu2.forward(self.hidden_layer2.forward(self.x_out_1))
        self.x_out_3 = self.final_layer.forward(self.x_out_2)
        out = self.softmax.forward(self.x_out_3)
        return out

    def backward(self, loss_back):
        
        self.dz3 = loss_back                                                # N x 10
        self.dW3, self.db3, da2 = self.final_layer.backward(self.dz3)       # 128 x 10, 1 x 10, N x 128
        self.dz2 = self.relu2.backward(da2)                                 # N x 128
        self.dW2, self.db2, da1 = self.hidden_layer2.backward(self.dz2)     # 256 x 128, 1 x 128, N x 256
        self.dz1 = self.relu1.backward(da1)                                 # N x 256
        self.dW1, self.db1, dX = self.hidden_layer1.backward(self.dz1)      # 784 x 256, 1 x 256