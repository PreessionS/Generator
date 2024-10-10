import torch
from diffusers import FluxPipeline

# Inicjalizacja FluxPipeline z uwierzytelnieniem
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    use_auth_token="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"  # Zastąp swoim rzeczywistym tokenem
)

# Włączenie przeniesienia na CPU, aby oszczędzić VRAM
pipe.enable_model_cpu_offload()

# Ustawienie promptu i parametrów generowania
prompt = "Kot trzymający tabliczkę z napisem hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

# Zapisanie wygenerowanego obrazu
image.save("flux-dev.png")
