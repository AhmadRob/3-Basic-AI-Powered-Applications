import cv2 # for image processing
import numpy as np # for numerical operations
import streamlit as st # for web app interface
from tensorflow.keras.applications.mobilenet_v2 import ( # MobileNetV2 model
    
    # Keras contains a lot of pre-trained models and utilities

    MobileNetV2, # for image classification
    preprocess_input, # for preprocessing input images
    decode_predictions # for decoding model predictions
)
from PIL import Image # for image handling

def load_model():
    # MobileNetV2 is a lightweight model suitable for mobile and edge devices.
    # MobileNetV2 is a convolutional neural network designed for efficient image classification.
    model = MobileNetV2(weights="imagenet") # Load the pre-trained MobileNetV2 model
    return model

def preprocess_image(image):
    img = np.array(image) # Convert PIL image to NumPy array
    img = cv2.resize(img, (224, 224)) # Resize image to 224x224 pixels
    img = preprocess_input(img) # Preprocess the image for MobileNetV2
    img = np.expand_dims(img, axis=0) # Add batch dimension
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image) # Preprocess the image
        predictions = model.predict(processed_image) # Get model predictions
        decoded_predictions = decode_predictions(predictions, top=3)[0] # Decode predictions

        return decoded_predictions # Return top 3 predictions
    except Exception as e:
        print(f"Error during image classification: {e}")
        return None
    
def main():
    st.set_page_config(page_title="Image Classification", page_icon="🖼️", layout="centered")
    
    st.title("AI Image Classifier")
    st.write("Upload an image and get predictions on what it contains.")

    @st.cache_resource
    def load_cached_model():    
        return load_model()
    
    model = load_cached_model() # Load the pre-trained model

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_column_width=True,
        )

        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Classifying..."):
                image = Image.open(uploaded_file) # Open the uploaded image
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions:")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score: .2f}")

if __name__ == "__main__":
    main()  # Run the main function when the script is executed