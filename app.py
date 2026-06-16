import streamlit as st
import os
from PIL import Image

# Define image folders
input_image_folder = "input images"  # Original JPG images
output_image_folder = "output images"  # Binary masks in PNG format

# Set page configuration
st.set_page_config(page_title="Skin Disease Discriminator", layout="wide")

# View images side by side like a discriminator process
st.title("Discriminator Visualization: Original vs Binary Mask")

for i in range(1, 63):
    original_image_path = os.path.join(input_image_folder, f"{i}.jpg")
    binary_mask_path = os.path.join(output_image_folder, f"{i}.png")

    if os.path.exists(original_image_path) and os.path.exists(binary_mask_path):
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image_path, caption=f"Original Image {i}", use_container_width=True)
        
        with col2:
            st.image(binary_mask_path, caption=f"Binary Mask {i}", use_container_width=True)

st.sidebar.markdown("🩺 **AI-Powered Discriminator View!** 🚀")