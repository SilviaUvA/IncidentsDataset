#CLEAN LABELS
import csv
import os
from PIL import Image, UnidentifiedImageError

# Paths to input and output CSV files
input_csv_file = "data/labels_eccv_train_more_low_classes.csv"
output_csv_file = "data/cleaned_labels_eccv_train_more_low_classes.csv"

# Directory containing the images
root_dir = "data/eccv_val_images"

# Open the input CSV and create an output CSV
with open(input_csv_file, 'r') as infile, open(output_csv_file, 'w', newline='') as outfile:
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)

    # Copy the header row to the new CSV file
    header = next(csv_reader)
    csv_writer.writerow(header)

    # Process each entry in the CSV
    for row in csv_reader:
        filename, label = row  # Assumes "filename, label" format
        img_path = os.path.join(root_dir, filename)

        try:
            # Attempt to open the image
            with Image.open(img_path) as img:
                img.verify()  # Verify that the image is valid
            csv_writer.writerow(row)  # Write valid row to the new CSV file
            # print(f"Image loaded successfully: {img_path}")
        except (UnidentifiedImageError, IOError) as e:
            # Skip writing to the new CSV and log the failed file
            print(f"Failed to load image: {img_path}, Error: {e}")
