import torch
import torch.nn as nn

class NeuralNetworkHyperspectral(nn.Module):
    def __init__(self, in_channels, mode="encoder"):
        super(NeuralNetworkHyperspectral, self).__init__()
        self.mode = mode
        self.layers = self._build_layers(in_channels)

    def _build_layers(self, in_channels):
        layers = []
        if self.mode == "encoder":
            layers += [
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ]
        elif self.mode == "decoder":
            layers += [
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
