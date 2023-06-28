from keras.utils import get_file
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from streamlit_drawable_canvas import st_canvas

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 0, 25, 1)
stroke_color = st.sidebar.color_picker("Stroke color : ")
fill_color = st.sidebar.color_picker("Fill color : ", "#fff")
bg_color = st.sidebar.color_picker("Background color : ", "#eee")

canvas_result = st_canvas(
    fill_color=fill_color,
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=256,
    width=256,
    drawing_mode=drawing_mode,
    key="canvas",
)

image_array = np.array(canvas_result.image_data)
image = image_array[:,:,:3]

if st.button("Generate"):
    pix2pix_path = get_file(
        origin="https://huggingface.co/CineAI/Pix2Pix/resolve/main/pix2pix.h5",
        file_hash="555576fa2ea8be3192c547a3355f4ac419ff9c2b94abcce5a2252b873055ce6b"
    )
    model = load_model(pix2pix_path)
    expand_dim = np.expand_dims(image, axis=0)
    generated = model(expand_dim)
    X = (generated + 1) / 2.0
    X = X * 255
    X = tf.cast(X, tf.uint8)
    st.image(np.squeeze(X))
