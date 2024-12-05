import json
import os
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

splits = ["val", "train"]
data_folder = "data"
img_path = f"{data_folder}/images"
os.makedirs(img_path, exist_ok=True)

for split in splits:
    # load json file
    with open(f"{data_folder}/multi_label_{split}.json", "r") as fp:
        dataset = json.load(fp)
    
    delete_keys = []

    # download images
    for img_name in tqdm(dataset.keys()): #TODO to resume downloading...
        url = dataset[img_name]["url"]

        # avoid double downloading
        split_img_name = img_name.split("/")
        if os.path.exists(f"{img_path}/{split_img_name[0]}_{split_img_name[1]}"):
            continue

        try:
            img_data = requests.get(url, timeout=10).content

            # sometimes urls work but no img anymore
            try:
                img = Image.open(BytesIO(img_data))
                img.verify()
            except (IOError, SyntaxError) as _:
                print(f"Invalid image data for {img_name} at {url}")
                delete_keys.append(img_name)
                continue
        except requests.exceptions.RequestException as _:
            # stale url or website did not respond quickly
            print(f"Failed to download {img_name} from {url}.")
            delete_keys.append(img_name)
            continue
        
        # store
        with open(f"{img_path}/{split_img_name[0]}_{split_img_name[1]}", "w+b") as handler:
            handler.write(img_data)

    # remove stale urls from dataset
    for key in delete_keys:
        del dataset[key]

    # save dataset with valid imgs only
    with open(f"{data_folder}/multi_label_{split}_clean.json", "w") as file:
        json.dump(dataset, file)