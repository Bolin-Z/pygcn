import numpy as np
from utils import glorot_init

class GraphConvolutionLayer:
    def __init__(self, n_inputs, n_outputs, activation=None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_init(self.n_outputs, self.n_inputs) # weight matrix (out, in)
        self.activation = activation
    
    def forward(self, A, X, W=None):
        # A (bs, bs) adjacent matrix bs = 'batch size'
        # X (bs, in) input features
        self._A = A
        self._X = (A @ X).T # (in, bs)

        if W is None:
            W = self.W
        
        H = W @ self._X # (out, in) * (in, bs) -> (out, bs)
        if self.activation is not None:
            H = self.activation(H)
        self._H = H # (out, bs)
        return self._H.T # (bs, out)
    
    def backward(self, optim, update=True):
        # suppose use tanh as activation
        dtanh = 1 - np.asarray(self._H.T) ** 2 # (bs, out)
        d2 = np.multiply(optim.out, dtanh)

        # df(x) / dx
        self.grad = self._A @ d2 @ self.W # (bs, bs) * (bs, out) * (out, in) -> (bs, in)
        optim.out = self.grad

        dW = np.asarray(d2.T @ self._X.T) / optim.bs # (out, bs) * (bs, in) -> (out, in)
        dW_wd = self.W * optim.wd / optim.bs # weight decay update

        if update:
            self.W -= (dW + dW_wd) * optim.lr
        
        return dW + dW_wd
    
    def reset_param(self):
        self.W = glorot_init(self.n_outputs, self.n_inputs)

class AffineSoftMaxLayer:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_init(self.n_outputs, self.n_inputs) # (out, in)
        self.b = np.zeros((self.n_outputs, 1)) # (out, 1)
        self._X = None

    def shift(self, proj):
        shiftx = proj - np.max(proj, axis=0, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=0, keepdims=True)
    
    def forward(self, X, W=None, b=None):
        # X: (bs, in)
        # affine layer
        self._X = X.T # (in, bs)
        if W is None:
            W = self.W # (out, in)
        if b is None:
            b = self.b # (out, 1)
        
        proj = np.asarray(W @ self._X) # (out, in) * (in, bs) = (out, bs)
        # softmax layer
        return self.shift(proj).T # (bs, out)
    
    def backward(self, optim, update=True):
        train_mask = np.zeros(optim.y_pred.shape[0])
        train_mask[optim.train_nodes] = 1
        train_mask = train_mask.reshape((-1, 1)) # (bs, 1)

        # dxent(x) / dx
        d1 = np.asarray((optim.y_pred - optim.y_true)) # (bs, out)
        # element-wise multiply
        d1 = np.multiply(d1, train_mask) # (bs, out)

        # daffine(x) / dx
        self.grad = d1 @ self.W # (bs, out) * (out, in) -> (bs, in)
        optim.out = self.grad

        dW = (d1.T @ self._X.T) / optim.bs # (out, bs) * (bs, in) -> (out, in)
        db = d1.T.sum(axis=1, keepdims=True) / optim.bs # (out, 1)

        dW_wd = self.W * optim.wd / optim.bs # weight decay update

        if update:
            self.W -= (dW + dW_wd) * optim.lr
            self.b -= db.reshape(self.b.shape) * optim.lr
        
        return dW + dW_wd, db.reshape(self.b.shape)
    
    def reset_param(self):
        self.W = glorot_init(self.n_outputs, self.n_inputs)
        self.b = np.zeros((self.n_outputs, 1))