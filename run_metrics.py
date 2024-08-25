# 1. Image -> Embedding. Similarity to the Origin Image.
# 2. Image -> Keywords.  Keywords similarity to the Origin Image.
import os
import json

from tqdm import tqdm
from loguru import logger
from PIL import Image, ImageFile

import torch
from torch import nn, norm
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    Blip2Processor,
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPProcessor,
    CLIPModel,
)


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
        local_files_only=False,
        bit4_config=None,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.device = device
        self.max_length = max_length
        self.tokenizer = CLIPTokenizer.from_pretrained(
            version,
            local_files_only=local_files_only,
            use_fast=True,
        )
        self.transformer = CLIPTextModel.from_pretrained(
            version,
            local_files_only=local_files_only,
            # quantization_config=bit4_config,
        ).to(self.device)
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text, *args, **kwargs):
        return self(text)


class ClipMetric:
    def __init__(self, device="cuda:1") -> None:
        # init the clip model
        blip2_model_path = "Salesforce/blip2-opt-2.7b"
        logger.info("begin to load blip")
        self.device = device
        self.blip_processor = Blip2Processor.from_pretrained(blip2_model_path)
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            blip2_model_path,
            device_map={"": device},
            local_files_only=True,
        )
        logger.info("begin to load clip")
        # self.clip_model = FrozenCLIPEmbedder(local_files_only=True, device=device)
        self.clip_model = CLIPModel.from_pretrained(
            "/Ckpts/clip-vit-base-patch32", local_files_only=True
        ).to(self.device)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            "/Ckpts/clip-vit-base-patch32",
            local_files_only=True,
            use_fast=True,
        )
        self.clip_text_model = CLIPTextModel.from_pretrained(
            "/Ckpts/clip-vit-base-patch32",
            local_files_only=True,
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained("/Ckpts/clip-vit-base-patch32")
        # clip text
        self.max_length = 77
        self.layer = "last"

    def _encode(self, text):
        batch_encoding = self.clip_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.clip_text_model(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        z = outputs.last_hidden_state
        return z

    def _convert_img_caption(self, img):
        inputs = self.blip_processor(images=img, return_tensors="pt").to(self.device)
        output = self.blip_model.generate(**inputs)
        res = self.blip_processor.decode(output[0], skip_special_tokens=True)
        return res

    def _convert_img_emb(self, img):
        caption = self._convert_img_caption(img)
        emb = self._encode(caption)
        return emb

    def embeddingSim(self, origin, target):
        inputs = self.processor(
            images=[origin, target], return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        cosine_similarity = F.cosine_similarity(
            image_features[0], image_features[1], dim=0
        )
        return cosine_similarity

    def keywordsSim(self, origin, target):
        origin_emb = self._convert_img_emb(origin)
        target_emb = self._convert_img_emb(target)
        # logger.info(f"origin_emb: {origin_emb.size()}")
        return F.cosine_similarity(origin_emb, target_emb, dim=-1)

    def calEmbeddingSim(self, origin, normal, gen):
        normal_sim = self.embeddingSim(origin, normal)
        gen_sim = self.embeddingSim(origin, gen)
        return normal_sim, gen_sim

    def calKeywordsSim(self, origin, normal, gen):
        normal_sim = self.keywordsSim(origin, normal)
        gen_sim = self.keywordsSim(origin, gen)
        return normal_sim, gen_sim


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    prefix_path = "./"
    dict_path = "my_dict.jsonl"
    img_path = os.path.join(prefix_path, "results")
    origin_path = "/ssd/sdf/lllrrr/Datasets/POG_sd/"

    loaded_dict = {}
    metric_dict = {}

    # 从 JSONL 文件加载字典
    with open(os.path.join(prefix_path, dict_path), "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            loaded_dict.update(entry)

    logger.info(f"len of loaded dict: {len(loaded_dict)}")

    # loop the dict
    logger.info("begin to load the model")
    clip_metric = ClipMetric()
    logger.info("begin to process the data")
    for key, value in tqdm(loaded_dict.items()):
        # load the images
        origin_img = Image.open(os.path.join(origin_path, key))
        normal_img = Image.open(os.path.join(img_path, value["normal"]))
        lora_img = Image.open(os.path.join(img_path, value["lora"]))
        # cal sim
        (emb_normal_sim, emb_lora_sim) = clip_metric.calEmbeddingSim(
            origin_img, normal_img, lora_img
        )
        (key_normal_sim, key_lora_sim) = clip_metric.calKeywordsSim(
            origin_img, normal_img, lora_img
        )
        if key not in metric_dict.keys():
            metric_dict[key] = {}
            metric_dict[key]["emb"] = {}
            metric_dict[key]["key"] = {}
        metric_dict[key]["emb"]["normal"] = emb_normal_sim.item()
        metric_dict[key]["emb"]["lora"] = emb_lora_sim.item()
        metric_dict[key]["key"]["normal"] = key_normal_sim.mean().item()
        metric_dict[key]["key"]["lora"] = key_lora_sim.mean().item()
        logger.info(
            f"emb_lora_sim: {emb_lora_sim.item()}, emb_normal_sim: {emb_normal_sim.item()}"
        )
        logger.info(
            f"key_lora_sim: {key_lora_sim.mean().item()}, key_normal_sim: {key_normal_sim.mean().item()}"
        )

    # Save the dict to jsonl
    with open("metric_dict.jsonl", "w") as f:
        for key, value in metric_dict.items():
            json_line = json.dumps({key: value})
            f.write(json_line + "\n")
