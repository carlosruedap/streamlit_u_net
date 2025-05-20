import numpy as np
import streamlit as st
from PIL import Image
from src.yolo import predict, get_result, preprocessing, predict_colonies
from src.utils import adjust_mask
import cv2

st.set_page_config(
    page_title="Urosario",
    page_icon="🧫",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.write("## SIBIO - 🧫")

# File uploader
image = st.file_uploader("Cargar la imagen para detectar pozos", type=["png", "jpg", "jpeg", "tiff"])

# Session state for storing image and intermediate results
if 'img' not in st.session_state:
    st.session_state.img = None
if 'mask' not in st.session_state:
    st.session_state.mask = None
if 'mask_refined' not in st.session_state:
    st.session_state.mask_refined = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'individual_mask' not in st.session_state:
    st.session_state.individual_mask = None
if 'index_mask' not in st.session_state:
    st.session_state.index_mask = None

if image is not None:
    st.session_state.mask = None
    st.session_state.mask_refined = None
    st.session_state.result = None
    st.session_state.individual_mask = None
    st.session_state.index_mask = None

    # Read the uploaded image using OpenCV
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    st.session_state.img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #st.session_state.img = Image.open(image)
    print(f'La imagen cargada es: {st.session_state.img}')
    # st.image(st.session_state.img, caption='Imagen cargada', use_column_width=True)

    st.session_state.mask, st.session_state.individual_mask, st.session_state.index_mask = predict(st.session_state.img)
    st.session_state.mask_refined = adjust_mask(st.session_state.mask)
    st.session_state.result = get_result(np.float32(st.session_state.mask_refined), st.session_state.img)
    print("MAX AND MIN:", np.max(st.session_state.result), np.min(st.session_state.result))
    # st.image(st.session_state.result, caption='Resultado', use_column_width=True)
    # cv2.imwrite("image.jpg", st.session_state.result)


if st.session_state.result is not None:
    st.image(st.session_state.result, caption='Resultado', use_column_width=True)  # Visualización solo de los pozos

    threshold1 = st.slider("threshold 1", 0, 100, 20)
    threshold2 = st.slider("threshold 2", 101, 180, 180)

    processed_result = preprocessing(st.session_state.result, threshold1, threshold2)
    st.image(processed_result, caption='Resultado', use_column_width=True)

    on = st.toggle("Detectar desde imagen original", value=True)
    detect_colonies = st.button("Detectar colonias", type="primary", use_container_width=True)

    if detect_colonies:
        if on:
            colonies, colony_counts, total_colonies, col_img = predict_colonies(st.session_state.img, st.session_state.individual_mask)
            # colonies = adjust_mask(colonies)
            colonies = get_result(np.float32(colonies), st.session_state.img)
            # print(colon)
            print("MAX AND MIN 2:", np.max(colonies), np.min(colonies))
            st.image(colonies, caption='Colonies', use_column_width=True)
            # for i, col in enumerate(col_img):
                # st.image(col / 255.0, caption=f'Pozo {i+1}', use_column_width=True)
        else:
            colonies, colony_counts, total_colonies, col_img = predict_colonies(processed_result, st.session_state.individual_mask)
            # colonies = adjust_mask(colonies)
            colonies = get_result(np.float32(colonies), st.session_state.img)
            # print(colon)
            print("MAX AND MIN 2:", np.max(colonies), np.min(colonies))
            st.colonies,image( caption='Colonies', use_column_width=True)
            # for i, col in enumerate(col_img):
               # st.image(col / 255.0, caption=f'Pozo {i+1}', use_column_width=True)

        st.write("### Número de colonias por pozo:")
        for i in range(1, 7):
            pos = st.session_state.index_mask.index(i)
            if pos != -1:
                st.write(f"Pozo {i}: {colony_counts[pos]} colonias")

        st.write(f"### Número de colonias: {total_colonies}")
