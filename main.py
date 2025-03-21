import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load trained models
try:
    model1 = tf.keras.models.load_model('C:\\Users\\Saiku\Downloads\\Projects GitHub\\Weapons Classification custom and Vgg19\\vgg19_multilabel_model_sample.h5')
    model2 = tf.keras.models.load_model('C:\\Users\\Saiku\\Downloads\\Projects GitHub\\Weapons Classification custom and Vgg19\\binary_cnn_model.h5')
except FileNotFoundError:
    st.error("Trained models not found. Ensure both model files are in the specified directory.")
    st.stop()

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file, target_size):

    try:
        image = Image.open(uploaded_file)
        image = image.resize(target_size)
        image = np.array(image) / 255.0  # Normalize pixel values
        if len(image.shape) == 2:  # If grayscale, convert to RGB
            image = np.stack([image] * 3, axis=-1)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit app
st.title("Image Classification with Two Models")

# File upload
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Add a button to trigger prediction
    if st.button("Predict"):
        # Preprocess the image for both models
        image_model1 = preprocess_image(uploaded_file, target_size=(224, 224))  # For VGG19
        image_model2 = preprocess_image(uploaded_file, target_size=(64, 64))  # For custom model

        if image_model1 is not None and image_model2 is not None:
            # Make predictions with both models
            prediction1 = model1.predict(image_model1)
            prediction2 = model2.predict(image_model2)

            # Get predicted class indices
            class_index1 = np.argmax(prediction1)
            class_index2 = np.argmax(prediction2)

            # Define class labels for both models
            class_labels = ['Automatic Rifle','Bows or Arrows','Knifes','Short Gun',
                            'Sniper', 'Spears or Polearms', 'SubMachine Gun','Sword','pistols']
            # Display predictions in two columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<h1 style='color:blue;'>Model 1 (VGG19) Prediction</h1>", unsafe_allow_html=True)
                st.write(f"Predicted Class: {class_labels[class_index1]}", style={'font-size': '20px'})

            with col2:
                st.markdown("<h1 style='color:green;'>Model 2 (Custom CNN) Prediction</h1>", unsafe_allow_html=True)
                st.write(f"Predicted Class: {class_labels[class_index2]}", style={'font-size': '20px'})
