import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model (fixed space in path)
model_path = "danish_model_2.h5"
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(180, 180))  # Adjust size if needed
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

# Streamlit app
def main():
    st.title("Image Classification App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(uploaded_file)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Get class labels (replace with your actual class labels)
        class_labels = ["class1", "class2", "class3"]  # Example

        # Display the predicted class and probabilitys
        st.write("Predicted Class:", class_labels[predicted_class])
        st.write("Probability:", predictions[0][predicted_class])

if __name__ == "__main__":
    main()