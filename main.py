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

def train_model(dataloader, learning_rate=0.001, num_epochs=10):
    """Trains the model with the given parameters and returns the final loss."""
    model = CompressionModel(in_channels=204)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

    return model, loss.item()  # Return final model and loss

def hyperparameter_validation(dataset_dir, learning_rates, batch_sizes, num_epochs=10):
    """Iterates through learning rates and batch sizes, training and evaluating each combination."""
    best_loss = float('inf')
    best_params = None
    best_model = None

    dataset = HyperspectralDatasetLoader(dataset_dir)
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with learning rate = {lr} and batch size = {batch_size}")
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Train model with current hyperparameters
            model, final_loss = train_model(dataloader, learning_rate=lr, num_epochs=num_epochs)

            # Check if current configuration gives better loss
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                best_model = model
            print(f"Loss: {final_loss:.4f} for learning rate = {lr}, batch size = {batch_size}")

    print(f"Best loss: {best_loss:.4f} with parameters: {best_params}")
    return best_model, best_params  # Return best model and best hyperparameters

def compress_and_decompress_image(model, image_tensor):
    """Compresses and decompresses an image using the trained model."""
    with torch.no_grad():
        compressed = model.encoder(image_tensor.unsqueeze(0)).squeeze().cpu().numpy()
        reconstructed = model(image_tensor.unsqueeze(0)).squeeze().cpu().numpy()

        original_image = image_tensor.cpu().numpy().transpose(1, 2, 0)
        reconstructed_image = reconstructed.transpose(1, 2, 0)
        compressed_image = compressed.transpose(1, 2, 0)

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(original_image[..., 0], cmap="gray")
        axes[0].set_title("Original Image")
        axes[1].imshow(compressed_image[..., 0], cmap="gray")
        axes[1].set_title("Compressed Image")
        axes[2].imshow(reconstructed_image[..., 0], cmap="gray")
        axes[2].set_title("Reconstructed Image")
        plt.show()

def compress_and_decompress_images(model, images_tensor):
    for image in images_tensor:
        compress_and_decompress_image(model, image_tensor=image)

# Example usage
if __name__ == "__main__":
    dataset_path = "dataset_hs_nov11"

    # Hyperparameter ranges to test
    learning_rates = [0.001, 0.0001, 0.00001]
    batch_sizes = [4, 8, 16]
    
    
    
    #я заранил валидацию с одним эпохом, посмотрел какие были лучшие значения и вписал их сюда
    learning_rates = [0.00001] 
    batch_sizes = [16] 
    
    # Run hyperparameter validation
    best_model, best_params = hyperparameter_validation(
        dataset_dir=dataset_path,
        learning_rates=learning_rates,
        batch_sizes=batch_sizes,
        num_epochs=10
    )

    print(f"Best hyperparameters found: {best_params}")

    # Save the best model
    save_model(best_model, save_path="best_model_weights.pkl")



        #Вот эта строчка для того чтобы тренить модельку с нуля
    #model = train_compression_model(dataset_path)
    #model = train_model(dataloader, learning_rate=lr, num_epochs=num_epochs)

    #вот эти две строчки чтобы загрузить модельку которую уже натренили
    #model = CompressionModel(in_channels=204)
    #load_model(model)

    #вот это чтобы юзать натренированую модельку
    example_image = torch.rand((204, 64, 64))  # Replace with actual image tensor
    dataset = HyperspectralDatasetLoader(dataset_path)
    image_0 = dataset.__getitem__(0)
    image_1 = dataset.__getitem__(1)
    image_2 = dataset.__getitem__(2)

    model = best_model

    # Stack images into a NumPy array for testing
    array = dataset.__gettensor__()
    compress_and_decompress_images(model, array)