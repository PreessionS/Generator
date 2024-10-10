import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch

@st.cache_resource
def load_model():
    pipeline = AutoPipelineForText2Image.from_pretrained(
        'black-forest-labs/FLUX.1-schnell', 
        torch_dtype=torch.bfloat16
    )
    pipeline.load_lora_weights('hugovntr/flux-schnell-realism', weight_name='schnell-realism_v1')
    return pipeline

st.title("FLUX.1-schnell Image Generator")

pipeline = load_model()

prompt = st.text_input("Enter your prompt:", "Moody kitchen at dusk, warm golden light")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = pipeline(prompt).images[0]
    st.image(image, caption="Generated Image", use_column_width=True)
