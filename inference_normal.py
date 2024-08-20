from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, local_files_only=True
)
pipe.to("cuda")

# prompt = "a woman in a red and black dress"
# prompt = "a black polka dot dress"
prompt = "a red handbag with a bow"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
prompt = prompt.replace(" ", "_")
image.save("results/" + prompt + "_normal.png")
