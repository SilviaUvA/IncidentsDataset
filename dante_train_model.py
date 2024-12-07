import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm  # For progress bars
import os
import pandas as pd
from dante_parser import get_parser
import pprint
import warnings
from dante_metrics import AverageMeter, compute_ap_for_top1
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from dante_utils import eval

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
    train_csv_file = "data/filtered_labels_val_copy.csv"
    root_dir = "data/eccv_val_images"
    model_save_dir = "models/train_full/"
    eval_csv_file = "data/test_eval_val.csv"

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
    dataset = LargeImageDataset(csv_file=train_csv_file, root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # Use a ResNet model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),          # hidden layer
    nn.ReLU(),   
    nn.Linear(1024, 44)                             # Output layer
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # Training parameters
    num_epochs = 10

    # Track and trace training stats
    mean_ap_per_epoch = []
    class_ap_per_epoch = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels, idx in tqdm(dataloader, desc="Training Batches"):
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

        # Epoch loss
        epoch_loss = running_loss / len(dataloader)
        print(f"Loss: {epoch_loss:.4f}")

        # Run Validation
        print(f"Running evaluation on validation data for this epoch!")
        mean_ap, per_class_ap = eval(model, eval_csv_file)

        # Epoch mAP
        # Save mAP and per_class_AP 
        mean_ap_per_epoch.append(mean_ap)
        class_ap_per_epoch.append(per_class_ap)
        print(f"Per class AP: {per_class_ap}")
        print(f"Mean Average Precision (mAP): {mean_ap:.4f}")

        # Save the model for this epoch
        checkpoint_path = os.path.join(model_save_dir, f"model_full_train_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    print("Training complete!")

    # Save stats
    mean_ap_per_epoch_np = np.array(mean_ap_per_epoch)
    np.save("results/train_full/mean_ap_per_epoch", mean_ap_per_epoch_np)

    class_ap_per_epoch_np = np.array(class_ap_per_epoch)
    print(class_ap_per_epoch_np.shape)
    np.save("results/train_full/class_ap_per_epoch", class_ap_per_epoch_np)

if __name__ == "__main__":
    main()
