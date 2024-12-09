import csv
from collections import defaultdict
import math
import random

# Path to your CSV file
train_csv_path = "data/perma_train_labels.csv"

# Dictionary to hold labels as keys and lists of filenames as values
train_dict = defaultdict(list)

# Read the CSV file
with open(train_csv_path, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header if there is one

    # Populate the dictionary
    for row in reader:
        filename, label = row
        train_dict[label].append(filename)

# Convert defaultdict to a regular dict (optional)
train_dict = dict(train_dict)

# Function to write a dictionary to a CSV file
def write_dict_to_csv(file_path, label_to_filenames):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])  # Write the header

        # Loop over labels and filenames
        for label, filenames in label_to_filenames.items():
            for filename in filenames:
                writer.writerow([filename, label])  # Write each filename-label pair

# # Train/Eval Write to csv
# write_dict_to_csv(train_csv_path, train_dict)
# write_dict_to_csv(eval_csv_path, eval_dict)



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Split train dict further  into labeled / unlabeled data for active learning
active_base_labeled_csv_path = "data/active_base_labeled_1000.csv"
active_base_unlabeled_csv_path = "data/active_base_unlabeled_1000.csv"

labeled_dict = {}
unlabeled_dict = {}

num_each_incident = 10
total_base_samples = 1000
num_classes = 43

for label, filenames in train_dict.items():
    # print(f"Label: {label}, Files: {len(filenames)}")
    num_total = len(filenames)

    if int(label) == 0:
        num_labeled = total_base_samples - (num_classes * num_each_incident)
    else:
        num_labeled = num_each_incident 

    shuffled_list = random.sample(filenames, num_total)

    labeled_dict[label] = shuffled_list[:num_labeled]
    unlabeled_dict[label] = shuffled_list[num_labeled:]

# Write to active_base_labeled and active_base_unlabeled csv's
write_dict_to_csv(active_base_labeled_csv_path, labeled_dict)
write_dict_to_csv(active_base_unlabeled_csv_path, unlabeled_dict)