# app.py
import os
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import streamlit as st

# Ustaw token Hugging Face jako zmienną środowiskową
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Wklej swój token zamiast hf_your_token_here

# Nazwa modelu
model_name = "black-forest-labs/FLUX.1-dev"

# Inicjalizacja pipeline z modelem FLUX.1
pipe = DiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    use_auth_token=True
)
pipe.enable_model_cpu_offload()  # Wymagane dla oszczędności pamięci VRAM

# Aplikacja Streamlit
st.title("FLUX.1 Image Generator")
prompt = st.text_input("Wpisz opis obrazu:", value="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")

if st.button("Generuj obraz"):
    with st.spinner("Generowanie obrazu..."):
        # Generowanie obrazu na podstawie promptu
        image = pipe(
            prompt,
            height=512,  # Możesz dostosować wysokość
            width=512,   # Możesz dostosować szerokość
            guidance_scale=7.5,
            num_inference_steps=50
        ).images[0]

        # Wyświetlanie obrazu
        st.image(image, caption="Wygenerowany obraz", use_column_width=True)
