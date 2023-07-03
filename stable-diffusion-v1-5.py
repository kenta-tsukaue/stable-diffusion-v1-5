from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


file_name = input("ファイル名を教えてね\n")
prompt = input("どんな画像が欲しいかを教えてね\n")
image = pipe(prompt).images[0]  
    
image.save(file_name + ".png")
