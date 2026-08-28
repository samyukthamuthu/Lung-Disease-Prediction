import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lung Disease Prediction",
    page_icon="🫁",
    layout="centered"
)

# ============================================================
# TITLE
# ============================================================

st.title("🫁 Lung Disease Prediction")
st.write(
    "Upload a chest X-ray image to predict the lung condition."
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "lung_disease_final.keras"
    )

    return model


try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the prediction model.")
    st.error(str(e))
    st.stop()

# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Fibrosis",
    "Normal",
    "Pneumonia"
]

# ============================================================
# SUGGESTIONS
# ============================================================

suggestions = {

    "Fibrosis":
        "Suggestion: The image shows features associated with Fibrosis. "
        "Consider reviewing the result with a qualified healthcare professional "
        "for further evaluation.",

    "Normal":
        "Suggestion: The image appears consistent with a Normal chest X-ray. "
        "Continue regular health monitoring and seek medical advice if symptoms persist.",

    "Pneumonia":
        "Suggestion: The image shows features associated with Pneumonia. "
        "Consider consulting a qualified healthcare professional for further evaluation."
}

# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        width="stretch"
    )

    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    img = image.resize(
        (224, 224)
    )

    img_array = np.array(
        img
    ).astype(
        np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    with st.spinner("Analyzing X-ray..."):

        prediction = model.predict(
            img_array,
            verbose=0
        )

    predicted_class = np.argmax(
        prediction[0]
    )

    predicted_name = class_names[
        predicted_class
    ]

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Condition: {predicted_name}"
    )

    # --------------------------------------------------------
    # SUGGESTION
    # --------------------------------------------------------

    st.subheader("Suggestion")

    st.info(
        suggestions[predicted_name]
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Lung Disease Prediction System | "
    "MobileNetV2 + EfficientNetB0 Hybrid Model"
)