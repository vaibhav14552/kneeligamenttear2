import numpy as np
import streamlit as st
import cv2
from PIL import Image

# Custom CSS for styling
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #87CEEB; /*sky blue */
        color: #000;
        font-family: sans-serif;
    }
    .custom-heading {
        font-family: 'serif'; /* Or any other nice font like 'Georgia', 'Times New Roman' */
        font-size: 2.5em;
        color: #333; /* Black text */
        background-color:#f0f8ff ; /* light blue background */
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 5px 5px 10px #888888; /* Optional shadow */
    }
    .subheader {
        color: #4682b4; /* Steel Blue */
        margin-top: 20px;
    }
    .info-box {
        background-color: #333; /* dark gray */
        color: #4682b4; /* steel blue text */
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid #add8e6; /* Light Sky Blue */
    }
    .error-box {
        background-color: #ffe0e0;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid #ffb3b3;
        color: #a94442;
    }
    .success-box {
        background-color: #fffacd;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid #b3ffb3;
        color: #3c763d;
    }
    .warning-box {
        background-color: #fffacd; /* Lemon Chiffon */
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid #eee8aa; /* Dark Khaki */
        color: #8a6d3b;
    }
    .image-container {
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .caption {
        font-size: 0.9em;
        color: #555;
        text-align: center;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='custom-heading'>Knee Ligament Detection</h1>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>Please upload exactly two MRI images of the knee for ligament detection. The system is specifically designed to analyze MRI scans.</div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload Two MRI Images of the Knee",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Enhanced MRI detection logic
def is_mri_like(img):
    if img is None or img.size == 0:
        return False

    # Check grayscale dominance
    b, g, r = cv2.split(img)
    diff = (np.abs(b - g).mean() + np.abs(g - r).mean() + np.abs(b - r).mean()) / 3
    grayscale_like = diff < 15

    # Check edge density
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    detailed_structure = edge_density > 0.025

    # Check for specific texture characteristics
    try:
        from skimage.feature import local_binary_pattern
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        texture_presence = np.any(hist > 0.01)
    except ImportError:
        texture_presence = True

    return grayscale_like and detailed_structure and texture_presence

if uploaded_files and len(uploaded_files) == 2:
    valid_mri_images = []
    for i, uploaded_file in enumerate(uploaded_files):
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is not None and is_mri_like(image):
            valid_mri_images.append(image)
        else:
            st.markdown(f"<div class='error-box'>Image {i + 1} does not appear to be a valid MRI image. Please upload MRI knee scans only.</div>", unsafe_allow_html=True)

    if len(valid_mri_images) == 2:
        for i, image in enumerate(valid_mri_images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            detection_image = image.copy()
            ligament_detected = False

            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 10000:
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-7)
                    if 0.1 < circularity < 0.8:
                        aspect_ratio = 0
                        x, y, w, h = cv2.boundingRect(contour)
                        if h > 0:
                            aspect_ratio = float(w) / h
                        if 0.2 < aspect_ratio < 5:
                            ligament_detected = True
                            cv2.drawContours(detection_image, [contour], -1, (0, 255, 0), 2)

            st.markdown(f"<h3 class='subheader'>MRI Image {i + 1}</h3>", unsafe_allow_html=True)
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(image, caption=f"Original MRI Image {i + 1}", channels="BGR")
            st.markdown("<p class='caption'>Original Image</p></div>", unsafe_allow_html=True)

            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(edges, caption=f"Edge Detection {i + 1}")
            st.markdown("<p class='caption'>Edge Detection</p></div>", unsafe_allow_html=True)

            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(detection_image, caption=f"Ligament Detection Result {i + 1}", channels="BGR")
            st.markdown("<p class='caption'>Ligament Detection Result</p></div>", unsafe_allow_html=True)

            if ligament_detected:
                st.markdown("<div class='success-box'>Ligament-like structures detected in the MRI image.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='warning-box'>No clear ligament-like structures detected in the MRI image.</div>", unsafe_allow_html=True)
    elif valid_mri_images:
        st.markdown("<div class='warning-box'>Please upload exactly two valid MRI images of the knee.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='info-box'>Please upload exactly two MRI images of the knee.</div>", unsafe_allow_html=True)