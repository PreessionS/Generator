import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Załaduj model Diffusers
model_name = "runwayml/stable-diffusion-v1-5"  # Możesz wybrać inny model dostępny w diffusers
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # Upewnij się, że model działa na CPU

# Interfejs aplikacji
st.title("Generator obrazów z Diffusers na Streamlit")

# Wprowadzanie promptu
prompt = st.text_input("Wprowadź opis obrazu:")

# Generowanie obrazu
if st.button("Generuj obraz"):
    if prompt:
        with st.spinner("Generowanie obrazu..."):
            image = pipe(prompt).images[0]  # Generowanie obrazu na podstawie promptu
            st.image(image, caption="Wygenerowany obraz", use_column_width=True)
    else:
        st.write("Wprowadź opis obrazu, aby wygenerować obraz.")
