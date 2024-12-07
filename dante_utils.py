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

    for images, labels in tqdm(dataloader, desc="Evaluation Batches"):
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
