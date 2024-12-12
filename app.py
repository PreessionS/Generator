# app.py
import streamlit as st
import torch
from PIL import Image
import os
from huggingface_hub import login

# Konfiguracja środowiska
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['HF_HOME'] = '/tmp/hf_home'

# Konfiguracja strony
st.set_page_config(page_title="FLUX Image Generator", layout="wide")

# Funkcja logowania do HuggingFace
def init_huggingface():
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = ""
    
    st.sidebar.title("HuggingFace Authentication")
    token_input = st.sidebar.text_input(
        "Enter HuggingFace Token",
        value=st.session_state.hf_token,
        type="password"
    )
    
    if token_input != st.session_state.hf_token:
        st.session_state.hf_token = token_input
        if token_input:
            try:
                login(token=token_input)
                st.sidebar.success("Successfully logged in!")
                return True
            except Exception as e:
                st.sidebar.error(f"Login failed: {str(e)}")
                return False
    return bool(st.session_state.hf_token)

# Funkcja ładowania modelu
@st.cache_resource
def load_model():
    try:
        from diffusers import StableDiffusionPipeline
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
        
        # Ładowanie wag LoRA
        pipe.load_lora_weights("lexa862/NSFWmodel")
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            st.sidebar.success("Using GPU acceleration")
        else:
            st.sidebar.warning("Running on CPU (this will be slow)")
        
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Full error:", exc_info=True)
        return None

# Główna funkcja generowania obrazu
def generate_image(pipe, prompt, num_steps=20):
    try:
        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=num_steps,
            ).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        st.error("Full error:", exc_info=True)
        return None

def main():
    st.title("FLUX Image Generator")
    
    # Sprawdzenie uwierzytelnienia
    if not init_huggingface():
        st.warning("Please enter your HuggingFace token in the sidebar to continue.")
        return
    
    # Ładowanie modelu
    if 'model' not in st.session_state:
        with st.spinner('Loading model...'):
            st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("Failed to load model")
        return
    
    # Interface użytkownika
    col1, col2 = st.columns([2, 1])
    
    with col2:
        num_steps = st.slider("Number of inference steps", 10, 50, 20)
        
    with col1:
        prompt = st.text_area("Enter your prompt:", height=100)
        
    if st.button('Generate Image', type='primary'):
        if not prompt:
            st.warning("Please enter a prompt")
            return
            
        with st.spinner('Generating image...'):
            image = generate_image(st.session_state.model, prompt, num_steps)
            if image is not None:
                st.image(image, caption=f"Generated image for: {prompt}")
                
                # Opcja pobrania
                if st.button('Download Image'):
                    # Konwersja do bytes
                    import io
                    buf = io.BytesIO()
                    image.save(buf, format='PNG')
                    st.download_button(
                        label="Download PNG",
                        data=buf.getvalue(),
                        file_name="generated_image.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()
