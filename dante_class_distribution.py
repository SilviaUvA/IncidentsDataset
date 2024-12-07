import csv
import numpy as np

csv_file_path = 'data/filtered_labels_val_copy.csv'

label_dict = {}
for i in range(44):
    label_dict[i] = 0


print(label_dict)

with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    
    # Loop through each row in the CSV file
    for row in reader:
        # Ensure the row has at least two columns
        if len(row) > 1:
            label_dict[int(row[1])] += 1
   
print(label_dict)