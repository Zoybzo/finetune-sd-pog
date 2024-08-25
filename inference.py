import os
import json
import shutil
import re

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from loguru import logger

from diffusers import StableDiffusionPipeline


# the file path
prefix_path = "/ssd/sdf/lllrrr/Datasets/POG_sd/"
metadata_name = "metadata.csv"
output_dir = "./results/"
image_name_file = "./images.csv"
# get the csv
metadata = pd.read_csv(os.path.join(prefix_path, metadata_name))
image_names = pd.read_csv(image_name_file)
# process dict
process_dict = metadata.set_index("file_name").to_dict("index")
# check the image name
image_names_remove = image_names["file_name"].tolist()
process_dict = {
    key: value for key, value in process_dict.items() if key not in image_names_remove
}
metadata = metadata[~metadata["file_name"].isin(image_names_remove)]
logger.info(f"len of process_dict: {len(process_dict)}")
# add the key 'normal' and the key 'lora' to the value in each dict
# for key, value in process_dict.items():
#     value["normal"] = "default_value_for_normal"
#     value["lora"] = "default_value_for_lora"


class PogDataset(Dataset):
    def __init__(self, file) -> None:
        super().__init__()
        self.metadata = file

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        # get the line of the metadata by the index
        row = self.metadata.iloc[index]
        return {"img": row.file_name, "prompt": row.text}


dataset = PogDataset(metadata)
# Assuming `dataset` is a custom dataset class that yields prompts
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)


# lora sd
def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)


cp_num = "5000"
model_path = "/ssd/sdf/lllrrr/Projects/sd/sd-pog-model/checkpoint-" + cp_num
lora_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, local_files_only=True
)
lora_pipe.unet.load_attn_procs(model_path)
lora_pipe.safety_checker = dummy_safety_checker
lora_pipe.to("cuda:1")
# normal sd
normal_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, local_files_only=True
)
normal_pipe.safety_checker = dummy_safety_checker
normal_pipe.to("cuda:1")


def sanitize_filename(filename):
    # 定义正则表达式，匹配不适合文件名的字符（包括空格、特殊符号等）
    sanitized_filename = re.sub(r'[\/:*?"<>|]', "_", filename)
    return sanitized_filename


# process
for batch in dataloader:
    file_names = batch["img"]
    prompts = batch["prompt"]
    # images
    normal_images = [normal_pipe(prompt).images[0] for prompt in prompts]
    lora_images = [lora_pipe(prompt).images[0] for prompt in prompts]
    logger.info(type(normal_images))
    # Save each image in the batch
    for i, image in enumerate(normal_images):
        file_name = file_names[i]
        image_name = prompts[i].replace(" ", "_") + "_normal.png"
        image_name = sanitize_filename(image_name)
        # add the image_name to dict
        process_dict[file_name]["normal"] = image_name
        output_path = os.path.join(output_dir, f"{image_name}")
        image.save(output_path)
    for i, image in enumerate(lora_images):
        file_name = file_names[i]
        image_name = prompts[i].replace(" ", "_") + "_lora.png"
        image_name = sanitize_filename(image_name)
        process_dict[file_name]["lora"] = image_name
        output_path = os.path.join(output_dir, f"{image_name}")
        image.save(output_path)
    for i, prompt in enumerate(prompts):
        file_name = os.path.join(prefix_path, file_names[i])
        image_name = prompt.replace(" ", "_") + "_origin.png"
        image_name = sanitize_filename(image_name)
        target = os.path.join(output_dir, image_name)
        shutil.copy(file_name, target)
    with open(image_name_file, "a") as f:
        for file_name in file_names:
            f.write(file_name + "\n")
    break
# 保存字典到 JSONL 文件
with open("my_dict.jsonl", "w") as f:
    for key, value in process_dict.items():
        json_line = json.dumps({key: value})
        f.write(json_line + "\n")
