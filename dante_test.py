import pandas as pd
import os
from PIL import Image, ImageFile

# Allow Pillow to handle truncated images without crashing
ImageFile.LOAD_TRUNCATED_IMAGES = False

# File paths
csv_file_path = "data/cleaned_labels_eccv_val.csv"
root_dir = "data/eccv_val_images"
output_csv_path = "data/filtered_labels.csv"

# Read the CSV
data_frame = pd.read_csv(csv_file_path)

# Create a list to hold valid entries
valid_entries = []

# Loop through each row in the CSV
for index, row in data_frame.iterrows():
    img_name = os.path.join(root_dir, row['filename'])
    try:
        # Attempt to open and fully load the image
        with Image.open(img_name) as img:
            img.load()  # Fully load the image to detect any truncation issues
        valid_entries.append(row)  # Add to valid entries if no exception is raised
    except OSError as e:
        # Log the problematic file
        print(f"Skipping corrupted or truncated image: {img_name}, Error: {e}")

# Save valid entries to a new CSV
filtered_data_frame = pd.DataFrame(valid_entries, columns=data_frame.columns)
filtered_data_frame.to_csv(output_csv_path, index=False)

print(f"Filtered CSV saved to {output_csv_path}")
