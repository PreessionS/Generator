# app.py
import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import os

def load_model():
    # Inicjalizacja modelu
    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Załadowanie wag LoRA
    pipe.load_lora_weights("xey/sldr_flux_nsfw_v2-studio")
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

def generate_image(pipe, prompt):
    # Generowanie obrazu
    image = pipe(prompt).images[0]
    return image

def main():
    st.title("Generator obrazów AI")
    
    # Inicjalizacja modelu przy pierwszym uruchomieniu
    if 'model' not in st.session_state:
        with st.spinner('Ładowanie modelu...'):
            st.session_state.model = load_model()
    
    # Interface użytkownika
    prompt = st.text_area("Wprowadź opis obrazu:", 
                         "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")
    
    if st.button('Generuj obraz'):
        with st.spinner('Generowanie obrazu...'):
            try:
                image = generate_image(st.session_state.model, prompt)
                st.image(image, caption='Wygenerowany obraz')
            except Exception as e:
                st.error(f"Wystąpił błąd podczas generowania: {str(e)}")

if __name__ == "__main__":
    main()
