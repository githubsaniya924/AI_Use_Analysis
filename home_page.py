import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# ------------------- CUSTOM CSS -------------------
st.markdown(
    """
    <style>
    @keyframes floatAnimation {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .floating-image {
        animation: floatAnimation 3s ease-in-out infinite;
        width: 120px; /* Adjust width */
        height: auto;
        margin: 10px;
    }

    .image-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
    }

    .stApp {
        background-image: url("https://source.unsplash.com/1600x900/?ai,technology");
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""This dataset explores AI adoption trends among students and professionals, covering 50,000 records with 25 attributes.
             It provides insights into demographics, AI usage patterns, efficiency improvements, and challenges faced in AI integration.

Using data visualization, hypothesis testing and clustering techniques like K-Means, we uncover trends in AI adoption and its influence on work and learning.""")

# ------------------- FUNCTION TO LOAD & ENCODE IMAGE -------------------
def load_and_encode_image(image_path):
    image = Image.open(image_path).resize((120, 120))  # Resize for better UI
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ------------------- FLOATING AI TOOLS IMAGES -------------------
#st.subheader("AI Tools in Use")
ai_images = ["chatgpt.jpeg", "copilot.jpeg", "grammarly.jpeg","midjourney.jpeg"]  # Add more if needed
image_html = ""

for img in ai_images:
    try:
        img_base64 = load_and_encode_image(img)
        image_html += f'<img class="floating-image" src="data:image/png;base64,{img_base64}" /> '
    except FileNotFoundError:
        pass  # Skip missing images

st.markdown(f'<div class="image-container">{image_html}</div>', unsafe_allow_html=True)




# ------------------- APP FOOTER -------------------
st.markdown("---")
st.markdown("**AI & Data Analysis Dashboard** | Developed with using Streamlit")