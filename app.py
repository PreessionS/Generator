import streamlit as st
from diffusers import DiffusionPipeline
from PIL import Image
import torch

def main():
    st.title("Generator obrazów AI")
    st.write("Użyj modelu FLUX.1 do generowania obrazów")

    # Input dla promptu
    prompt = st.text_area("Wprowadź prompt:", 
                         "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                         height=100)

    # Przycisk do generowania
    if st.button("Generuj obraz"):
        with st.spinner("Ładowanie modelu i generowanie obrazu..."):
            try:
                # Inicjalizacja pipeline
                pipe = DiffusionPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    torch_dtype=torch.float16
                )
                
                # Przeniesienie na GPU jeśli dostępne
                if torch.cuda.is_available():
                    pipe = pipe.to("cuda")
                
                # Załadowanie dodatkowych wag LoRA
                pipe.load_lora_weights("xey/sldr_flux_nsfw_v2-studio")
                
                # Generowanie obrazu
                image = pipe(prompt).images[0]
                
                # Wyświetlenie wygenerowanego obrazu
                st.image(image, caption=prompt, use_column_width=True)
                
                # Opcja pobrania
                buf = Image.new(mode="RGB", size=image.size)
                buf.paste(image)
                st.download_button(
                    label="Pobierz obraz",
                    data=buf,
                    file_name="wygenerowany_obraz.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Wystąpił błąd podczas generowania: {str(e)}")

if __name__ == "__main__":
    main()
