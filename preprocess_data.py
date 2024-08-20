import os
import json
from tqdm import tqdm
import pandas as pd

data_dir = "/ssd/sdf/lllrrr/Datasets/POG"
file_name = "item_data_keywords.txt"
result_dir = "/ssd/sdf/lllrrr/Datasets/POG_sd"

data = pd.read_csv(os.path.join(data_dir, file_name), sep=",")
# get the item_id and img_caption columns
data = data[["item_id", "img_caption"]]
# 将DataFrame转换为JSON Lines格式并写入文件
# save the data as csv file with header ['file_name', 'text']
data.columns = ["file_name", "text"]
data["file_name"] = data["file_name"].astype(str) + ".png"
data.to_csv(os.path.join(result_dir, "metadata.csv"), index=False)
print("DataFrame已成功存储为csv 格式文件。")
