import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Inicjalizacja modelu
model_id = "runwayml/stable-diffusion-v1-5"  # To jest otwarty model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Jeśli masz GPU, użyj go dla lepszej wydajności
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Funkcja do generowania obrazu
def generuj_obraz(prompt, nazwa_pliku="wygenerowany_obraz.png", seed=None):
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    image = pipe(prompt, generator=generator).images[0]
    image.save(nazwa_pliku)
    print(f"Obraz zapisany jako {nazwa_pliku}")
    return image

# Przykładowe użycie
prompt = "Kot siedzący na księżycu, styl artystyczny"
generuj_obraz(prompt, "kot_na_ksiezycu.png", seed=42)

# Możesz generować więcej obrazów, zmieniając prompt
prompt2 = "Futurystyczne miasto nocą, neony, deszcz"
generuj_obraz(prompt2, "futurystyczne_miasto.png")
