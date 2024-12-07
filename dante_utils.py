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
from dante_metrics import AverageMeter, compute_ap_for_top1
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
# Dataset class (use the previously defined LargeImageDataset)
from dante_dataset import LargeImageDataset  
import csv

# Suppress the specific UserWarning
warnings.filterwarnings(
    "ignore", 
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images"
)


def eval(model, csv_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "data/eccv_val_images"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],            
                                        std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Create the dataset and DataLoader
    dataset = LargeImageDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # Calculate the Average precision over the epoch
    num_classes = 44
    aps_cumulative = np.zeros(num_classes)          
    num_samples_cumulative = np.zeros(num_classes)  # To accumulate number of samples seen for each class

    model.eval()
    with torch.no_grad():
        for images, labels, idx in tqdm(dataloader, desc="Evaluation Batches"):
            # Move data to GPU if available
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Convert labels to one-hot encoding
            probs = torch.softmax(outputs, dim=1)

            y_true_bin = label_binarize(labels.cpu().numpy(), classes=np.arange(num_classes))
            y_pred_bin = probs.detach().cpu().numpy()

            # For each class, compute the average precision and update cumulative values
            for class_idx in range(num_classes):
                ap = average_precision_score(y_true_bin[:, class_idx], y_pred_bin[:, class_idx])
                aps_cumulative[class_idx] += ap
                num_samples_cumulative[class_idx] += 1

        # Eval mAP, mean and per class
        mean_ap = np.sum(aps_cumulative) / np.sum(num_samples_cumulative)
        per_class_ap = aps_cumulative / num_samples_cumulative

    return mean_ap, per_class_ap


# Function to compute uncertainty
def compute_uncertainty(outputs):
    probs = torch.softmax(outputs, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)  # Add small value to avoid log(0)
    return entropy


def eval_uncertainty(model, unlabeled_dataloader, active_temp_unlabeled, num_new_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "data/eccv_val_images"

    model.eval()
    uncertainties = []
    indices = []
    with torch.no_grad():
        for images, labels, idx in tqdm(unlabeled_dataloader, desc="Batch Eval Uncertainty"):
            # Move data to GPU if available
            images = images.to(device)

            outputs = model(images)

            uncertainty = compute_uncertainty(outputs)
            uncertainties.extend(uncertainty.cpu().numpy())
            indices.extend(idx.cpu().numpy())


        # Select x most uncertain samples
        uncertainties = np.array(uncertainties)
        indices = np.array(indices)
        uncertain_indices = indices[np.argsort(-uncertainties)[:num_new_samples]]

    return uncertain_indices


def copy_csv(base_csv, temp_csv):
    #OG, labbeled base
    with open(base_csv, 'r') as infile:
        reader = csv.reader(infile)
        # Read all lines into a list
        header = next(reader)
        lines = list(reader)

    # copy OG, Labeled base
    # Overwrite the original file with the same data
    with open(temp_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write all the lines to the new file
        writer.writerow(header)
        writer.writerows(lines)


def add_samples_csv(idx_new_samples, active_temp_labeled, active_temp_unlabeled):
    # From unlabeled.csv grab the new samples you found
    with open(active_temp_unlabeled, 'r') as infile:
        reader = csv.reader(infile)
        # Read all lines into a list
        header = next(reader)
        lines = list(reader)
        # Only take the selected new sample lines
        new_sample_lines = [lines[idx] for idx in sorted(idx_new_samples, reverse=True)]


    # Add them to the labeld.csv 
    with open(active_temp_labeled, 'r') as infile:
        reader = csv.reader(infile)
        # Read all lines into a list
        header = next(reader)
        lines = list(reader)
        new_header = header
        new_lines = lines + new_sample_lines
    # Save em by overwriting
    with open(active_temp_labeled, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write all the lines to the new file
        writer.writerow(new_header)
        writer.writerows(new_lines)


def remove_samples_csv(idx_new_samples, active_temp_unlabeled):
    # Remove the new sampled lines for unlabeled.csv
    with open(active_temp_unlabeled, 'r') as infile:
        reader = csv.reader(infile)
        # Read all lines into a list
        header = next(reader)
        lines = list(reader)
        # remove elemnts at the specified incides
        for index in sorted(idx_new_samples, reverse=True):
            del lines[index]
    new_header = header
    new_lines = lines
    # Save the reduced lines by overwriting
    with open(active_temp_unlabeled, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write all the lines to the new file
        writer.writerow(new_header)
        writer.writerows(new_lines)