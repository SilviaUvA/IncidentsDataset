import json
from tqdm import tqdm
import pprint
import requests
import os
import csv


def download_image(image_url, save_path, image_name, timeout=10):
    try:
        # Send a GET request to the image URL
        response = requests.get(image_url, stream=True, timeout=timeout)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Create full save_path
        full_save_path = os.path.join(save_path, f"{image_name}")

        # Open a file at the specified save path and write the content to it
        with open(full_save_path, 'wb') as file:
            for chunk in response.iter_content(1024):  # Write the content in chunks
                file.write(chunk)
        print(f"Image successfully downloaded: {full_save_path}")
        return True
    except requests.exceptions.Timeout:
        print(f"Request timed out while downloading {image_url}.")
        return False
    except Exception as e:
        print(f"Error occurred: {e}")
        return False


label_dict = {
 'no incident': 0,
 'damaged': 1,
 'flooded': 2,
 'dirty contamined': 3,
 'blocked': 4,
 'collapsed': 5,
 'snow covered': 6,
 'under construction': 7,
 'burned': 8,
 'on fire': 9,
 'with smoke': 10,
 'ice storm': 11,
 'drought': 12,
 'dust sand storm': 13,
 'thunderstorm': 14,
 'wildfire': 15,
 'tropical cyclone': 16,
 'heavy rainfall': 17,
 'tornado': 18,
 'derecho': 19,
 'earthquake': 20,
 'landslide': 21,
 'mudslide mudflow': 22,
 'rockslide rockfall': 23,
 'snowslide avalanche': 24,
 'volcanic eruption': 25,
 'sinkhole': 26,
 'storm surge': 27,
 'fog': 28,
 'hailstorm': 29,
 'dust devil': 30,
 'fire whirl': 31,
 'traffic jam': 32,
 'ship boat accident': 33,
 'airplane accident': 34,
 'car accident': 35,
 'train accident': 36,
 'bus accident': 37,
 'bicycle accident': 38,
 'motorcycle accident': 39,
 'van accident': 40,
 'truck accident': 41,
 'oil spill': 42,
 'nuclear explosion': 43
}

img_count = 0
save_path = "./data/eccv_val_images/"
os.makedirs(save_path, exist_ok=True)  # Create folder if it doesn't exist
download_json_file = "data/eccv_val.json"


csv_file_path = 'data/labels_eccv_val.csv'
# Check if the CSV file exists, if not, create it and write the header
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:  # Open in write mode to create the file
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])  # Write header


# Loop over all the url's in the json file
with open(download_json_file, "r") as fp:
    dataset = json.load(fp)

    for image_name in tqdm(dataset.keys()):
        # print(image_name)
        # pprint.pprint(dataset[image_name])
        # print(dataset[image_name]['incidents'])

        # Try to open and save the image
        image_url = dataset[image_name]["url"]
        succes = download_image(image_url, save_path, f"image{img_count}.jpg")

        # If download succesful
        if succes:
            # find label
            labels = dataset[image_name]["incidents"]

            # check if any label is 1 (true incident)
            has_value_1 = 1 in labels.values()

            if has_value_1:
                # check which incident was true
                for key, value in labels.items():
                    # If this is the case add the image name and corresponding numerical incident label to the csv file.
                    if value == 1:
                        with open(csv_file_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([f"image{img_count}.jpg", label_dict[key]])
                        # Only add 1 label of the first incident 
                        break
            # Add the image name and 0 label to the csv file meaning there is no incident in this image
            else:
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f"image{img_count}.jpg", 0])

        
            # Increment counter of saved images with 1 for proper saving.
            img_count += 1