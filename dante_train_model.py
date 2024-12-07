import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm  # For progress bars
import os
import pandas as pd
from dante_parser import get_parser
import pprint
import warnings
from metrics import AverageMeter, accuracy, validate


# Suppress the specific UserWarning
warnings.filterwarnings(
    "ignore", 
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images"
)

# Dataset class (use the previously defined LargeImageDataset)
from dante_dataset import LargeImageDataset  


def main():
    parser = get_parser()
    args = parser.parse_args()


    # Paths
    csv_file = "data/filtered_labels_val.csv"
    root_dir = "data/eccv_val_images"

    # Define data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],            
                                     std=[0.229, 0.224, 0.225])
    if args.train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # Create the dataset and DataLoader
    dataset = LargeImageDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Use a ResNet model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),  # First layer: in_features -> 1024
    nn.ReLU(),                             # Activation function
    )
    model.output_logits = nn.Linear(1024, 44)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in tqdm(dataloader, desc="Training Batches"):
            # Move data to GPU if available
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()  # Zero the parameter gradients
            loss.backward()
            optimizer.step()

            # Track the loss
            running_loss += loss.item()

            # incident_prec1 = accuracy(outputs, labels, topk=1)
            # incident_prec5 = accuracy(outputs, labels, topk=5)

            # print(f"BATCH: Top 1, accuracy: {incident_prec1}, Top 5, accuracy: {incident_prec5}")

        # Epoch loss
        epoch_loss = running_loss / len(dataloader)
        print(f"Loss: {epoch_loss:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
