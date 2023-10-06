import os
import requests
import tensorflow_hub as hub
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Load the TensorFlow Hub model from the specified URL
model_url = "https://tfhub.dev/agripredict/disease-classification/1"
model = hub.load(model_url)

@app.get('/')
def fun():
    return {"hello": "welcome"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    try:
        # Read the uploaded file
        contents = await file.read()

        # Open the image using Pillow (PIL)
        pil_image = Image.open(io.BytesIO(contents))

        # Resize the image to match the input requirements of the TensorFlow Hub model
        pil_image = pil_image.resize((300, 300))  # Adjust the size as needed

        # Convert the image to a NumPy array
        image = np.array(pil_image)

        # Normalize the pixel values to the range [0, 1] and adjust preprocessing as needed

        # Ensure the image has the correct shape and data type according to the model's requirements
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        # Make predictions using the loaded model
        predictions = model(image)

        # Get the class labels (replace with the actual class labels for your model)
        class_labels = ['Tomato Healthy', 'Tomato Septoria Leaf Spot', 'Tomato Bacterial Spot', 'Tomato Blight', 'Cabbage Healthy', 'Tomato Spider Mite', 'Tomato Leaf Mold', 'Tomato_Yellow Leaf Curl Virus', 'Soy_Frogeye_Leaf_Spot', 'Soy_Downy_Mildew', 'Maize_Ravi_Corn_Rust', 'Maize_Healthy', 'Maize_Grey_Leaf_Spot', 'Maize_Lethal_Necrosis', 'Soy_Healthy', 'Cabbage Black Rot']  # Update with your labels

        # Get the predicted class label
        predicted_class = np.argmax(predictions[0])

        # Get the corresponding class name
        predicted_class_name = class_labels[predicted_class]

        return {"Predicted class": predicted_class_name}

    except Exception as e:
        return {"Error": f"Error decoding or processing the image: {e}"}, 400
