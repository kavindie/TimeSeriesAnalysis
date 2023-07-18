import torch
from torch.nn import Module, Linear, Sequential, LeakyReLU, Sigmoid


class DNN(Module):
    """ A simple deep neural network with LeakyReLu activation. """

    def __init__(self, t_in, f_cond, t_out, f_mod, hidden_layers=[16, 16], device=0):
        super().__init__()
        self.shape_in = (t_in, f_cond)
        self.shape_out = (t_out, f_mod)
        hidden_layers = [t_in*f_cond, *hidden_layers, t_out*f_mod]
        net = []
        for nin, nout in zip(hidden_layers[:-1], hidden_layers[+1:]):
            net.append(Linear(nin, nout))
            net.append(LeakyReLU())
        net.pop()  # remove last activation
        net = Sequential(*net)
        self.net = net
        self.to(device)

    def forward(self, x):
        assert x.shape[1:] == self.shape_in
        y = self.net(x.flatten(1))
        return y.reshape(-1, *self.shape_out)


class DNNClassifier(Module):
    """ A DNN classifier. Only diffrence from the DNN model is this has a sigmoid activation at the end. """

    def __init__(self, t_in, f_cond, t_out, f_mod, hidden_layers=[16, 16], device=0):
        super().__init__()
        self.shape_in = (t_in, f_cond)
        self.shape_out = (t_out, f_mod)
        hidden_layers = [t_in*f_cond, *hidden_layers, t_out*f_mod]
        net = []
        for nin, nout in zip(hidden_layers[:-1], hidden_layers[+1:]):
            net.append(Linear(nin, nout))
            net.append(LeakyReLU())
        net.pop()  # remove last activation
        net.append(Sigmoid())
        net = Sequential(*net)
        self.net = net
        self.to(device)

    def forward(self, x):
        assert x.shape[1:] == self.shape_in
        y = self.net(x.flatten(1))
        return y.reshape(-1, *self.shape_out)