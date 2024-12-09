import csv
import numpy as np

csv_file_path = 'data/filtered_labels_all.csv'

label_dict = {}
for i in range(44):
    label_dict[i] = 0


with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    
    # Loop through each row in the CSV file
    for row in reader:
        # Ensure the row has at least two columns
        if len(row) > 1:
            label_dict[int(row[1])] += 1
   
print(label_dict)

needed_classes = []
needed_classes_counts = {}

count = 0

for key, value in label_dict.items():
    if key == 0:
        continue

    count += value

    if value < 500:
        needed_classes.append(key)
        needed_classes_counts[key] = value


print(f"num incidents: {count}, num non incidents: {label_dict[0]}")