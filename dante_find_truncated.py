import warnings
import csv
from PIL import Image
import os

# Paths to input and output CSV files
input_csv_file = "data/cleaned_labels_eccv_train_more_low_classes.csv"
# Directory containing the images
root_dir = "data/eccv_val_images"

# Function to load image with warning handling
def load_image_with_warning(filepath):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Catch all warnings
        try:
            image = Image.open(filepath)
            image.load()  # Ensure the file is fully loaded
        except Exception as e:
            print(f"Error loading image: {e}")
            print(filepath)
            return None

        # Check if warnings were issued
        for warning in w:
            if "Truncated File Read" in str(warning.message):
                print(f"Warning: {warning.message} for file {filepath}")
                # Handle as needed, e.g., log or skip the file
        return image



# Open the input CSV and create an output CSV
with open(input_csv_file, 'r') as infile:
    csv_reader = csv.reader(infile)
    next(csv_reader, None)

        # Process each entry in the CSV
    for row in csv_reader:
        filename, label = row  # Assumes "filename, label" format
        img_path = os.path.join(root_dir, filename)

        # Test with a potentially problematic file
        img = load_image_with_warning(img_path)