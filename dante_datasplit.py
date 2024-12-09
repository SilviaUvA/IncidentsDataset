import csv
from collections import defaultdict
import math
import random

# Path to your CSV file
csv_file_path = "data/filtered_labels_all.csv"
train_csv_path = "data/perma_train_labels.csv"
eval_csv_path = "data/perma_eval_labels.csv"


# Dictionary to hold labels as keys and lists of filenames as values
label_to_filenames = defaultdict(list)

# Read the CSV file
with open(csv_file_path, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header if there is one

    # Populate the dictionary
    for row in reader:
        filename, label = row
        label_to_filenames[label].append(filename)

# Convert defaultdict to a regular dict (optional)
label_to_filenames = dict(label_to_filenames)

train_dict = {}
eval_dict = {}

# Splitting train / eval
for label, filenames in label_to_filenames.items():
    # print(f"Label: {label}, Files: {len(filenames)}")
    num_total = len(filenames)
    num_train = math.floor(num_total * 0.8)
    num_eval = num_total - num_train

    shuffled_list = random.sample(filenames, num_total)

    train_dict[label] = shuffled_list[:num_train]
    eval_dict[label] = shuffled_list[num_train:]


# Function to write a dictionary to a CSV file
def write_dict_to_csv(file_path, label_to_filenames):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])  # Write the header

        # Loop over labels and filenames
        for label, filenames in label_to_filenames.items():
            for filename in filenames:
                writer.writerow([filename, label])  # Write each filename-label pair

# Train/Eval Write to csv
write_dict_to_csv(train_csv_path, train_dict)
write_dict_to_csv(eval_csv_path, eval_dict)



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Split train dict further  into labeled / unlabeled data for active learning
active_base_labeled_csv_path = "data/active_base_labeled.csv"
active_base_unlabeled_csv_path = "data/active_base_unlabeled.csv"

labeled_dict = {}
unlabeled_dict = {}

for label, filenames in train_dict.items():
    # print(f"Label: {label}, Files: {len(filenames)}")
    num_total = len(filenames)

    # 50 of each incident = 50 x 43 = 2150 positive labels // 5000 - 2150 negative labels
    # to start off with 5k training samples in total
    if int(label) == 0:
        num_labeled = 5000 - (43 * 50)
    else:
        num_labeled = 50 

    shuffled_list = random.sample(filenames, num_total)

    labeled_dict[label] = shuffled_list[:num_labeled]
    unlabeled_dict[label] = shuffled_list[num_labeled:]

# Write to active_base_labeled and active_base_unlabeled csv's
write_dict_to_csv(active_base_labeled_csv_path, labeled_dict)
write_dict_to_csv(active_base_unlabeled_csv_path, unlabeled_dict)