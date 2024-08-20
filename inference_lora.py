from diffusers import StableDiffusionPipeline
import torch

cp_num = "15000"
model_path = "/ssd/sdf/lllrrr/Projects/sd/sd-pog-model/checkpoint-" + cp_num
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, local_files_only=True
)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

# prompt = "a black polka dot dress"
prompt = "a red handbag with a bow"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
# convert blank to under line
prompt = prompt.replace(" ", "_")
image.save("results/" + prompt + "_" + cp_num + "_lora.png")
