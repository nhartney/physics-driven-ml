from torch.nn import Module, Sequential, Linear, ReLU, Tanh


class PointNN(Module):
    """Build a simple toy nn-based model to deal with point data"""

    def __init__(self):
        super().__init__()

        self.nn_encoder = Sequential(Linear(4, 32),
                                    ReLU(True),
                                    Linear(32, 64),
                                    ReLU(True),
                                    Linear(64, 128),
                                    ReLU(True))

        self.nn_decoder = Sequential(Linear(128, 64),
                                    ReLU(True),
                                    Linear(64, 32),
                                    ReLU(True),
                                    Linear(32, 1))
        

    def forward(self, input_tensor):
    
        # CNN encoder-decoder
        z = self.nn_encoder(input_tensor)
        y = self.nn_decoder(z)

        return y
    

class SimplePointNN(Module):
    """Build an even simpler toy nn-based model to deal with point data for debugging purposes"""

    def __init__(self):
        super().__init__()

        self.nn_encoder = Sequential(Linear(4, 4),
                                    ReLU(True))

        self.nn_decoder = Sequential(Linear(4, 1))
        

    def forward(self, input_tensor):
    
        # CNN encoder-decoder
        z = self.nn_encoder(input_tensor)
        y = self.nn_decoder(z)

        return y