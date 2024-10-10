import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image

# Ładowanie modelu DiffusionPipeline
from diffusers import DiffusionPipeline
import torch

model_name = "black-forest-labs/FLUX.1-dev"
pipe = DiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    use_auth_token="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"
)

# Interfejs użytkownika Streamlit
st.title("Generator obrazów z Diffusers na Streamlit")

# Wprowadzanie promptu
prompt = st.text_input("Wprowadź opis obrazu:", "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")

# Przycisk do generowania obrazu
if st.button("Generuj obraz"):
    if prompt:
        with st.spinner("Generowanie obrazu..."):
            image = pipe(prompt).images[0]  # Generowanie obrazu na podstawie promptu
            st.image(image, caption="Wygenerowany obraz", use_column_width=True)
    else:
        st.write("Wprowadź opis obrazu, aby wygenerować obraz.")
