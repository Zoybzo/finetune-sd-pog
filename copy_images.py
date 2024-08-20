import os
from tqdm import tqdm
import pandas as pd


data_dir = "/ssd/sdf/lllrrr/Datasets/POG"
image_dir = "/ssd/sdf/lllrrr/Datasets/POG/images_se"
file_name = "item_data_keywords.txt"
result_dir = "/ssd/sdf/lllrrr/Datasets/POG_sd"

data = pd.read_csv(os.path.join(data_dir, file_name), sep=",")
# check whether the item_id in the image_foler
for item_id in tqdm(data["item_id"]):
    if not os.path.exists(os.path.join(image_dir, str(item_id) + ".png")):
        print(f"{item_id} not in the image folder.")
    else:
        # copy the image to the result folder
        os.system(
            f"cp {os.path.join(image_dir, str(item_id) + '.png')} {os.path.join(result_dir, str(item_id) + '.png')}"
        )
