import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LargeImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image names and labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.data_frame = pd.read_csv(csv_file)  # Read CSV file into a DataFrame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get image file name and label by the index
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])  # File path from CSV
        label = self.data_frame.iloc[idx, 1]  # Label from CSV

        # Load image
        image = Image.open(img_name).convert("RGB")

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        return image, label, idx