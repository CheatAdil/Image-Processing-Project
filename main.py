import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from CompressionModel import CompressionModel  # Your model file
from HyperspectralDatasetLoader import HyperspectralDatasetLoader  # Dataset loader

def train_compression_model(dataset_path, num_epochs=10, learning_rate=0.001, save_images=True):
    # Load the dataset
    dataset = HyperspectralDatasetLoader(dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = CompressionModel(204)
    criterion = nn.MSELoss()  # Using Mean Squared Error for loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, inputs)  # MSE between reconstructed and original
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print average loss per epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
        
        # Save compressed and decompressed images on the last epoch
        if epoch == num_epochs - 1 and save_images:
            with torch.no_grad():
                for i, inputs in enumerate(dataloader):
                    if i > 0:  # Only output one batch for simplicity
                        break

                    # Get compressed and reconstructed images
                    compressed = model.encoder(inputs)
                    reconstructed = model(inputs)

                    # Convert to numpy for visualization (adjust as necessary)
                    inputs_np = inputs[0].cpu().numpy()
                    compressed_np = compressed[0].cpu().numpy()
                    reconstructed_np = reconstructed[0].cpu().numpy()

                    # Plotting original, compressed, and reconstructed images
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(inputs_np[0], cmap='gray')  # Adjust channel for visualization
                    axes[0].set_title('Original Image')

                    # Visualize compressed representation (e.g., first component)
                    axes[1].imshow(compressed_np[0], cmap='gray')
                    axes[1].set_title('Compressed Representation')

                    axes[2].imshow(reconstructed_np[0], cmap='gray')
                    axes[2].set_title('Reconstructed Image')

                    for ax in axes:
                        ax.axis('off')
                    plt.show()

    print("Training complete!")
    return model


# Example usage
if __name__ == "__main__":
    dataset_path = "dataset_hs_nov11"
    model = train_compression_model(dataset_path)
