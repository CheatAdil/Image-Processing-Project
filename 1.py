from HyperspectralDatasetLoader import HyperspectralDatasetLoader

class MainRunner:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def main(self):
        try:
            # Initialize the dataset loader
            dataset_loader = HyperspectralDatasetLoader(self.dataset_path)
            
            # Process all images in the dataset
            all_images = dataset_loader.process_all_images()
            
            # Check if any images were loaded
            if all_images.size > 0:
                print("success")
            else:
                print("failure: no images found or loaded")

        except Exception as e:
            print(f"failure: {e}")

# Example usage
if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "dataset_hs_nov11"
    
    # Initialize and run the main class
    runner = MainRunner(dataset_path)
    runner.main()
