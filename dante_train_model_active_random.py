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
from dante_utils import eval, eval_uncertainty, copy_csv, add_samples_csv, remove_samples_csv, pick_random_samples
import csv

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
    base_train_labelled = "data/active_base_labeled_1000.csv"
    base_train_unlabelled = "data/active_base_unlabeled_1000.csv"
    root_dir = "data/eccv_val_images"
    model_save_dir = "models/active_random_1000_epoch20/"
    result_folder = "results/active_random_1000_epoch20/"
    eval_csv_file = "data/perma_eval_labels.csv"
    active_random_all_used_train_samples = "results/active_random_1000_epoch20/active_random_save_train_samples.csv"

    # Use these temporary csv's files to keep track of the labeled or unlabeld data
    # And iteratively increase them during the epoch based on top k (500) most uncertain samples
    active_temp_labeled = "data/active_temp_labeled.csv"
    active_temp_unlabeled = "data/active_temp_unlabeled.csv"
    # Copy into the temp files, such that we don't have to change the "true base" ones,
    # For keeping everything the same when reproducing.
    # Copy labeled base into active_temp_labeled
    copy_csv(base_train_labelled, active_temp_labeled)
    # Copy unlabeled base into active_temp_unlabeled
    copy_csv(base_train_unlabelled, active_temp_unlabeled)

    # Number of new active learning samples per epoch
    num_new_samples = 20

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
    num_epochs = 20

    # Track and trace training stats
    mean_ap_per_epoch = []
    class_ap_per_epoch = []

    # Training loop
    for epoch in range(num_epochs):
        if epoch == 0:
            # Select the base labeled dataset at the start
            labeled_dataset = LargeImageDataset(csv_file=base_train_labelled, root_dir=root_dir, transform=transform)
            labeled_loader = DataLoader(labeled_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
            # Select the base unlabeled dataset at the start 
            # unlabeled_dataset = LargeImageDataset(csv_file=base_train_unlabelled, root_dir=root_dir, transform=transform)
            # unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
        else:
            # Select the dataset with added samples by active learning mechanism
            labeled_dataset = LargeImageDataset(csv_file=active_temp_labeled, root_dir=root_dir, transform=transform)
            labeled_loader = DataLoader(labeled_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
            # Select the base unlabeled dataset at the start 
            # unlabeled_dataset = LargeImageDataset(csv_file=active_temp_unlabeled, root_dir=root_dir, transform=transform)
            # unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

        model.train()  # Set the model to training mode
        running_loss = 0.0

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels, idx in tqdm(labeled_loader, desc="Training Batches"):
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
        epoch_loss = running_loss / len(labeled_loader)
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
        checkpoint_path = os.path.join(model_save_dir, f"model_active_random_epoch_{epoch+1}_mAP_{mean_ap:.4f}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

        # Evaluate uncertainty on unlabeled data
        # Select num_new_samples random samples
        idx_new_samples = pick_random_samples(active_temp_unlabeled, num_new_samples)

        #TODO TESTING
        # Open the CSV file and count lines
        with open(active_temp_labeled, newline='') as file:
            reader = csv.reader(file)
            num_lines = sum(1 for row in reader)  # Count each row
            print(f"Number of rows in temp_labeled: {num_lines}")
        # Open the CSV file and count lines
        with open(active_temp_unlabeled, newline='') as file:
            reader = csv.reader(file)
            num_lines = sum(1 for row in reader)  # Count each row
            print(f"Number of rows in temp_unlabeled: {num_lines}")

        # Add new samples to active_temp_labeled
        add_samples_csv(idx_new_samples, active_temp_labeled, active_temp_unlabeled)
        # Remove new samples from active_temp_unlabeled
        remove_samples_csv(idx_new_samples, active_temp_unlabeled)

        #TODO TESTING
        # Open the CSV file and count lines
        with open(active_temp_labeled, newline='') as file:
            reader = csv.reader(file)
            num_lines = sum(1 for row in reader)  # Count each row
            print(f"Number of rows in temp_labeled: {num_lines}")
        # Open the CSV file and count lines
        with open(active_temp_unlabeled, newline='') as file:
            reader = csv.reader(file)
            num_lines = sum(1 for row in reader)  # Count each row
            print(f"Number of rows in temp_unlabeled: {num_lines}")


    print("Training complete!")

    # Save stats
    mean_ap_per_epoch_np = np.array(mean_ap_per_epoch)
    np.save(f"{result_folder}mean_ap_per_epoch", mean_ap_per_epoch_np)

    class_ap_per_epoch_np = np.array(class_ap_per_epoch)
    np.save(f"{result_folder}class_ap_per_epoch", class_ap_per_epoch_np)

    # Save CSV file that has all data samples used during active training
    copy_csv(active_temp_labeled, active_random_all_used_train_samples)

if __name__ == "__main__":
    main()
