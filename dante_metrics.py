
import os
import torch
import time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
import warnings

# Suppress the specific UserWarning
warnings.filterwarnings(
    "ignore", 
    message="No positive class found in y_true, recall is set to one for all thresholds"
)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_ap_for_top1(y_true, y_pred, num_classes=44):
    """
    Compute Average Precision for the top-1 predicted class.

    :param y_true: The true labels (batch_size, num_classes)
    :param y_pred: The predicted probabilities (batch_size, num_classes)
    :return: The Average Precision for top-1
    """
    # For top-1, we take the highest predicted class for each sample
    top1_preds = torch.argmax(y_pred, dim=1).cpu().numpy()  # Predicted class index for each sample
    top1_true = y_true.detach().cpu().numpy()    # True class index for each sample

    # Initialize list for AP calculations
    aps = []

    # Loop over each class and compute AP
    for class_idx in range(num_classes):
        # Create binary labels for this class
        y_true_bin = (y_true == class_idx).int()
        y_pred_bin = (torch.from_numpy(top1_preds) == class_idx).int()

        # Compute average precision for this class
        ap = average_precision_score(y_true_bin, y_pred_bin)
        aps.append(ap)

    return np.mean(aps)  # Return mean AP across all classes