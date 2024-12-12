import streamlit as st
from diffusers import DiffusionPipeline

# Funkcja do generowania obrazów
@st.cache_resource
def load_pipeline():
    st.info("Loading model. This may take some time...")
    pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
    pipe.load_lora_weights("xey/sldr_flux_nsfw_v2-studio")
    return pipe

pipe = load_pipeline()

# Interfejs użytkownika
st.title("AI Image Generator")
st.write("Generate images using the FLUX.1-dev model!")

prompt = st.text_input("Enter your prompt:", "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
        st.success("Image generated successfully!")
