# import the necessary packages
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "cancernet_model.keras"
CLASS_NAMES = ["benign", "malignant"]


@st.cache_resource
def get_model():
	return load_model(MODEL_PATH)


def preprocess_image(image):
	image = image.convert("RGB").resize((48, 48))
	image = np.array(image).astype("float32") / 255.0
	return np.expand_dims(image, axis=0)


st.set_page_config(page_title="Breast Cancer Histology Classifier", layout="centered")
st.title("Breast Cancer Histology Classifier")
st.write(
	"Upload one or more 50x50 histology image patches (PNG/JPG) to classify "
	"them as benign or malignant."
)

uploaded_files = st.file_uploader(
	"Choose image(s)",
	type=["png", "jpg", "jpeg"],
	accept_multiple_files=True,
)

if uploaded_files:
	model = get_model()

	st.header("Results")
	for uploaded_file in uploaded_files:
		image = Image.open(uploaded_file)
		processed = preprocess_image(image)

		preds = model.predict(processed)[0]
		predicted_idx = int(np.argmax(preds))
		predicted_class = CLASS_NAMES[predicted_idx]
		confidence = preds[predicted_idx] * 100

		col1, col2 = st.columns([1, 2])
		with col1:
			st.image(image, caption=uploaded_file.name, use_container_width=True)
		with col2:
			st.subheader(f"{predicted_class.capitalize()} ({confidence:.2f}%)")
			st.write(f"Benign: {preds[0] * 100:.2f}%")
			st.write(f"Malignant: {preds[1] * 100:.2f}%")
		st.divider()
