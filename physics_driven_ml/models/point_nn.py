from torch.nn import Module, Sequential, Linear, ReLU, Tanh, Sigmoid


class PointNN(Module):
    """
    Build a simple toy nn-based model to deal with point data. The network takes 3
    inputs and returns one output. It has 4 hidden layers, each followed by a ReLU
    activation function.
    """

    def __init__(self):
        super().__init__()

        self.nn_encoder = Sequential(Linear(4, 5),
                                     ReLU(True),
                                    #  Sigmoid(),
                                     Linear(5, 5),
                                     ReLU(True),
                                    #  Sigmoid(),
                                     Linear(5, 5),
                                     ReLU(True))
                                    #   Sigmoid())

        self.nn_decoder = Sequential(Linear(5, 5),
                                     ReLU(True),
                                    #  Sigmoid(),
                                     Linear(5, 5),
                                     ReLU(True),
                                    #  Sigmoid(),
                                     Linear(5, 1),
                                    #  ReLU())
                                    #  Tanh())
                                     Sigmoid())


    def forward(self, input_tensor):

        # CNN encoder-decoder
        z = self.nn_encoder(input_tensor)
        y = self.nn_decoder(z)

        return y
