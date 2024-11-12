import torch.nn as nn
from NeuralNetworkHyperspectral import NeuralNetworkHyperspectral

class CompressionModel(nn.Module):
    def __init__(self, in_channels):
        super(CompressionModel, self).__init__()
        self.encoder = NeuralNetworkHyperspectral(in_channels=in_channels, mode="encoder")
        self.decoder = NeuralNetworkHyperspectral(in_channels=in_channels, mode="decoder")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
