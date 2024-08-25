from diffusers import StableDiffusionPipeline
import torch

cp_num = "3000"
model_path = "/ssd/sdf/lllrrr/Projects/sd/sd-pog-model/checkpoint-" + cp_num
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, local_files_only=True
)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda:2")

# prompt = "a black polka dot dress"
prompt = "white leather sneakers. You need to make sure your output is complete"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
# convert blank to under line
prompt = prompt.replace(" ", "_")
image.save("results/" + prompt + "_" + cp_num + "_lora_prompt.png")
