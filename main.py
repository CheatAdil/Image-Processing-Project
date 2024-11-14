import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import spectral as spy
from HyperspectralDatasetLoader import HyperspectralDatasetLoader
from CompressionModel import CompressionModel

def train_compression_model(dataset_dir, num_epochs=10, learning_rate=0.001):
    # Load dataset
    dataset = HyperspectralDatasetLoader(dataset_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model and optimizer
    model = CompressionModel(in_channels=204)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Output the results of the final batch (compressed and decompressed image)
    with torch.no_grad():
        original = batch[0].cpu().numpy().transpose(1, 2, 0)  # Convert back to HxWxC for visualization
        reconstructed_image = reconstructed[0].cpu().numpy().transpose(1, 2, 0)

        # Plot original and reconstructed images side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original Image")
        axes[1].imshow(reconstructed_image, cmap="gray")
        axes[1].set_title("Reconstructed Image")
        plt.show()
    
    return model

# Example usage
if __name__ == "__main__":
    dataset_path = "dataset_hs_nov11"
    model = train_compression_model(dataset_path)