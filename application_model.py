from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Define the path to your model and image directory
model_path = "danish_model_2.h5"
image_dir = "usr_dir"

# Load the pre-trained model
model = load_model(model_path)

# Get a list of image files in the directory
image_files = os.listdir(image_dir)

# Iterate through each image file
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)

    # Load the image and preprocess it
    img = image.load_img(image_path, target_size=(180, 180))  # Adjust image size if needed
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize the image

    # Perform predictions using the model
    predictions = model.predict(x)

    # Get the predicted class and probability
    predicted_class = np.argmax(predictions)
    predicted_probability = predictions[0][predicted_class]

    # Print the results for the current image
    print("Image:", image_file)
    print("Predicted class:", predicted_class)
    print("Predicted probability:", predicted_probability)
    print()