import os
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from HyperspectralDatasetLoader import HyperspectralDatasetLoader
from CompressionModel import CompressionModel

def save_model(model, save_path="trained_model_weights.pkl"):
    """Saves the model weights using pickle."""
    with open(save_path, 'wb') as f:
        pickle.dump(model.state_dict(), f)
    print(f"Model weights saved at: {save_path}")

def load_model(model, load_path="trained_model_weights.pkl"):
    """Loads the model weights from a pickle file."""
    with open(load_path, 'rb') as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)
    print(f"Model weights loaded from: {load_path}")

def train_compression_model(dataset_dir, num_epochs=10, learning_rate=0.001):
    """Trains and saves the model weights."""
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

    # Save the final model weights after training completes
    save_model(model)
    return model

def compress_and_decompress_image(model, image_tensor):
    """Compresses and decompresses an image using the trained model."""
    with torch.no_grad():
        # Compress and decompress the image
        compressed = model.encoder(image_tensor.unsqueeze(0)).squeeze().cpu().numpy()
        reconstructed = model(image_tensor.unsqueeze(0)).squeeze().cpu().numpy()

        # Display original and reconstructed images
        original_image = image_tensor.cpu().numpy().transpose(1, 2, 0)
        reconstructed_image = reconstructed.transpose(1, 2, 0)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image[..., 0], cmap="gray")
        axes[0].set_title("Original Image")
        axes[1].imshow(reconstructed_image[..., 0], cmap="gray")
        axes[1].set_title("Reconstructed Image")
        plt.show()

# Example usage
if __name__ == "__main__":
    dataset_path = "dataset_hs_nov11"

    #Вот эта строчка для того чтобы тренить модельку с нуля
    #model = train_compression_model(dataset_path)

    #вот эти две строчки чтобы загрузить модельку которую уже натренили
    model = CompressionModel(in_channels=204)
    load_model(model)

    #вот это чтобы юзать натренированую модельку
    example_image = torch.rand((204, 64, 64))  # Replace with actual image tensor
    compress_and_decompress_image(model, example_image)
