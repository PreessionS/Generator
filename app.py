# app.py
import streamlit as st
import sys
import torch
from PIL import Image
import os
from huggingface_hub import login

# Konfiguracja środowiska przed importem diffusers
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['HF_HOME'] = '/tmp/hf_home'

# Pobierz token HF z secrets
def init_huggingface():
    try:
        hf_token = st.secrets["hf_token"]
        login(token=hf_token)
        st.success("Successfully logged in to Hugging Face")
    except Exception as e:
        st.error(f"Error logging in to Hugging Face: {str(e)}")
        st.stop()

# Inicjalizacja HF na starcie
init_huggingface()

try:
    from diffusers import DiffusionPipeline
    st.success("Successfully imported DiffusionPipeline")
except Exception as e:
    st.error(f"Error importing DiffusionPipeline: {str(e)}")
    st.stop()

def load_model():
    try:
        # Inicjalizacja modelu z obsługą błędów
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            cache_dir='/tmp/model_cache',
            use_auth_token=True  # Używamy uwierzytelniania
        )
        
        # Załadowanie wag LoRA
        pipe.load_lora_weights("xey/sldr_flux_nsfw_v2-studio")
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            st.warning("CUDA not available, using CPU (this will be slow)")
        
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_image(pipe, prompt):
    try:
        with torch.inference_mode():
            image = pipe(prompt).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def main():
    st.title("Generator obrazów AI")
    
    # Inicjalizacja modelu przy pierwszym uruchomieniu
    if 'model' not in st.session_state:
        with st.spinner('Ładowanie modelu...'):
            st.session_state.model = load_model()
            if st.session_state.model is None:
                st.error("Failed to load model")
                st.stop()
    
    # Interface użytkownika
    prompt = st.text_area(
        "Wprowadź opis obrazu:", 
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )
    
    if st.button('Generuj obraz'):
        if st.session_state.model is not None:
            with st.spinner('Generowanie obrazu...'):
                image = generate_image(st.session_state.model, prompt)
                if image is not None:
                    st.image(image, caption='Wygenerowany obraz')
        else:
            st.error("Model nie jest załadowany poprawnie")

if __name__ == "__main__":
    main()
