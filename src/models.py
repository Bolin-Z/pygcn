import numpy as np
from layers import GraphConvolutionLayer, AffineSoftMaxLayer

class GCNNetwork:
    def __init__(self, n_inputs, n_outputs, n_layers, hidden_size, activation, seed=0) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.hidden_sizes = hidden_size
        self.activation = activation

        np.random.seed(seed)

        self.layers = list()
        # input layer
        gcn_in = GraphConvolutionLayer(self.n_inputs, hidden_size[0], activation)
        self.layers.append(gcn_in)

        # hidden layers
        for layer in range(self.n_layers):
            gcn = GraphConvolutionLayer(self.layers[-1].W.shape[0], hidden_size[layer], activation)
            self.layers.append(gcn)
        
        # output layer
        sm_out = AffineSoftMaxLayer(hidden_size[-1], self.n_outputs)
        self.layers.append(sm_out)
    
    def embedding(self, A, X):
        H = X
        for layer in self.layers[:-1]:
            H = layer.forward(A, H)
        return np.asarray(H)
    
    def forward(self, A, X):
        H  = self.embedding(A, X)
        # softmax
        p = self.layers[-1].forward(H)
        return np.asarray(p)
    
    def reset_param(self):
        for layer in self.layers:
            layer.reset_param()