import torch
import torch.nn as nn
import torch.nn.functional as F

class neural_network_hyperspectral(nn.Module):
    def __init__(self, input_channels, compression_factor=16):
        super(neural_network_hyperspectral, self).__init__()
        
        # Encoder: Reduce spatial and spectral dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256 // compression_factor, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder: Upsample back to original dimensions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 // compression_factor, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Use Sigmoid if the output is normalized between 0 and 1
        )
    
    def forward(self, x):
        # Pass through encoder
        encoded = self.encoder(x)
        
        # Pass through decoder
        decoded = self.decoder(encoded)
        
        return decoded

# Example usage:
input_channels = 31  # Hypothetical number of spectral bands
compression_factor = 16
model = neural_network_hyperspectral(input_channels, compression_factor)

# Assume a random hyperspectral image input with shape (Batch, Channels, Height, Width)
input_image = torch.randn(1, input_channels, 128, 128)  # Example shape
output_image = model(input_image)

print("Input shape:", input_image.shape)
print("Output shape:", output_image.shape)
