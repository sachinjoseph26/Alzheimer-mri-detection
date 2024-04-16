import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_model():
    model = tf.keras.models.load_model('best_model_1.keras')
    return model

model = load_model()

# Streamlit webpage layout
st.title('Alzheimer MRI Image Prediction')
st.write('This application predicts the class of Alzheimer based on MRI images.')

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

def predict(image, model):
    image = image.resize((128, 128))  # Resize the image to match the model's expected input size
    image = np.array(image)

    if len(image.shape) == 2:  # Grayscale image
        image = np.stack((image,)*3, axis=-1)
    elif image.shape[2] == 1:  # Single channel
        image = np.concatenate([image] * 3, axis=-1)

    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model.predict(image)
    class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class= predict(image, model)
    st.write(f'Prediction: {predicted_class}')

# Plotting function
def plot_image(image, label):
    """Function to plot a single image."""
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')

