from diffusers import StableDiffusionPipeline
import torch

def textToImage():
    print(torch.cuda.is_available())
    # Verifica se CUDA está disponível para usar GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                    safety_checker = None,
                                                    requires_safety_checker = False,
                                                    torch_dtype=torch.float32).to(device)
    pipe.safety_checker = lambda images, clip_input: (images, False)
    prompt = ""
    image = pipe(prompt).images[0]

    image.save("sexye.png")
