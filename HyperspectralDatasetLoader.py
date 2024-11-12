import os
import torch
from torch.utils.data import Dataset
import spectral  # Ensure spectral is installed
import numpy as np

class HyperspectralDatasetLoader(Dataset):
    def __init__(self, root_dir):
        """
        Initializes the dataset loader by scanning the directory for hyperspectral images.

        Parameters:
        - root_dir (str): Path to the directory containing the hyperspectral image folders.
        """
        self.root_dir = root_dir
        self.image_paths = self._get_image_paths()  # List of all hyperspectral image paths

    def _get_image_paths(self):
        """
        Collects paths to all hyperspectral images in the specified root directory.

        Returns:
        - image_paths (list): List of full paths to each hyperspectral image.
        """
        image_paths = []
        for folder_name in os.listdir(self.root_dir):
            capture_path = os.path.join(self.root_dir, folder_name, "capture", f"{folder_name}.hdr")
            if os.path.exists(capture_path):
                image_paths.append(capture_path)
        return image_paths

    def __len__(self):
        """
        Returns the total number of hyperspectral images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves and loads a hyperspectral image as a tensor.

        Parameters:
        - idx (int): Index of the image to load.

        Returns:
        - tensor_image (Tensor): The hyperspectral image as a PyTorch tensor.
        """
        image_path = self.image_paths[idx]
        hyperspectral_image = spectral.open_image(image_path).load()

        # Convert the image to a PyTorch tensor with dtype float32
        tensor_image = torch.tensor(hyperspectral_image, dtype=torch.float32)

        # Ensure the tensor shape is [channels, height, width] (PyTorch convention)
        if tensor_image.ndimension() == 3:
            tensor_image = tensor_image.permute(2, 0, 1)  # Rearrange to [channels, height, width]

        return tensor_image
