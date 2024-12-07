import csv
import os
import warnings
from PIL import Image, ImageFile, UnidentifiedImageError

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths to input and output CSV files
input_csv_file = "data/labels_eccv_val.csv"
output_csv_file = "data/cleaned_labels_eccv_val.csv"

# Directory containing the images
root_dir = "data/eccv_val_images"

# Log warnings to track truncated file reads
def log_warning(message, filename):
    with open("warning_logs.txt", "a") as log_file:
        log_file.write(f"Warning: {message} | File: {filename}\n")
    print(f"Warning logged: {message} | File: {filename}")

# Custom warning filter to capture the file causing the warning
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    if "Truncated File Read" in str(message):
        log_warning(message, current_file)

# Apply the custom warning handler
warnings.showwarning = custom_warning_handler

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
        current_file = img_path  # Track the current file being processed

        try:
            # Attempt to open the image
            with Image.open(img_path) as img:
                img.verify()  # Verify that the image is valid
                image = Image.open(img_path).convert("RGB")
            csv_writer.writerow(row)  # Write valid row to the new CSV file
            print(f"Image loaded successfully: {img_path}")
        except (UnidentifiedImageError, OSError, IOError) as e:
            # Skip writing to the new CSV and log the failed file
            # print(f"Failed to load image: {img_path}, Error: {e}")
            continue
