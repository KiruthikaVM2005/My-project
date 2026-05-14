import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load pretrained model
model = MobileNetV2(weights="imagenet")

labels = ["Healthy Leaf", "Leaf Spot", "Powdery Mildew", "Rust"]

def predict_disease(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)

    index = np.argmax(preds) % len(labels)

    return labels[index]