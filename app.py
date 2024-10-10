import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Konfiguracja strony
st.set_page_config(page_title="Text to Image Generator", page_icon="🖼️", layout="wide")

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

# Tytuł aplikacji
st.title("🎨 Text to Image Generation with Stable Diffusion")

# Ładowanie modelu
with st.spinner("Loading model... This may take a few minutes."):
    pipe = load_model()

# Interfejs użytkownika
prompt = st.text_input("Enter your prompt:", "A beautiful sunset over a calm ocean")
num_images = st.slider("Number of images to generate", 1, 4, 1)
guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5, 0.5)
num_inference_steps = st.slider("Number of inference steps", 1, 100, 50)

if st.button("Generate Image(s)"):
    if prompt:
        try:
            with st.spinner("Generating image(s)... This may take a minute."):
                images = pipe(
                    [prompt] * num_images,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images

            cols = st.columns(num_images)
            for i, image in enumerate(images):
                cols[i].image(image, use_column_width=True, caption=f"Generated Image {i+1}")
                
                # Dodanie przycisku do pobrania obrazu
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                cols[i].download_button(
                    label="Download Image",
                    data=buf.getvalue(),
                    file_name=f"generated_image_{i+1}.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a prompt.")

# Dodanie informacji o aplikacji
st.markdown("---")
st.markdown("""
    This app uses the Stable Diffusion model to generate images from text prompts.
    It's powered by Hugging Face's Diffusers library and Streamlit.
    
    Note: This application requires significant computational resources and may not work properly without a GPU.
""")

# Dodanie stopki
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and Hugging Face Diffusers")
