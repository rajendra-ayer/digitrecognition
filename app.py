import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Digit Recognizer", layout="centered")

# -------------------------------
# Load trained MNIST model
# -------------------------------
@st.cache_resource
def load_digit_model():
    return load_model("mnist_cnn.h5")

model = load_digit_model()

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess(img):
    img = img.convert("L")  # grayscale
    arr = np.array(img)

    # Auto crop around digit
    threshold = 250
    mask = arr < threshold
    if mask.any():
        ys, xs = np.where(mask)
        arr = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
        img = Image.fromarray(arr)

    # Pad to square
    w, h = img.size
    size = max(w, h)
    new_img = Image.new("L", (size, size), 255)
    new_img.paste(img, ((size - w)//2, (size - h)//2))

    # Resize to 28×28
    new_img = new_img.resize((28, 28), Image.LANCZOS)

    # Invert if white background
    arr = np.array(new_img)
    if arr.mean() > 127:
        new_img = ImageOps.invert(new_img)

    arr = np.array(new_img).astype("float32") / 255.0
    arr = arr.reshape(28, 28, 1)
    return arr

# -------------------------------
# UI — Live Drawing Area
# -------------------------------
st.title("✏️ Live Handwritten Digit Recognition")
st.write("Draw any **digit (0–9)** below:")

canvas = st_canvas(
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas.image_data is not None:
    img = Image.fromarray(canvas.image_data.astype("uint8")).convert("RGB")
    st.image(img, width=200)

if st.button("Predict"):
    if canvas.image_data is None:
        st.error("Please draw a digit first!")
    else:
        x = preprocess(img)
        pred = model.predict(np.expand_dims(x, 0))
        digit = int(np.argmax(pred))
        conf = float(np.max(pred))

        st.success(f"**Prediction: {digit}** (confidence {conf:.2f})")
