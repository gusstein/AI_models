import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image

# Caminho para a imagem local
local_image_path = "../input/galaticos.jpg"
init_image = Image.open(local_image_path).convert("RGB")

def imageToImage():
    if init_image is None:
        print("Falha ao carregar a imagem.")
    else:
        print("Imagem carregada com sucesso.")

    # Inicializa a pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")

    # Define o prompt e outros parâmetros para a geração da imagem
    prompt = ("A highly detailed and realistic version of a football team logo, featuring a glossy black shield with silver"
              " swords and soccer ball, and bright stars above, all with realistic shadows and textures")
    strength = 0.75
    guidance_scale = 7.5
    print(type(init_image))
    # Carrega a imagem modificada pela pipeline
    try:
        result = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale)
        result_image = result.images  # Acesso correto às imagens resultantes
        result_image[0].save("generated_image.png")
        print("Imagem gerada e salva com sucesso.")
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")

