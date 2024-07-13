import os

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, send_from_directory
from flask import render_template, request

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('currency_detection_model.h5')

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Serve uploaded images from the 'uploads' directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

 
# Ensure the 'uploads' directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded image to the 'uploads' directory
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(image_path)
            print(image_path)

            # Preprocess the image
            processed_image = preprocess_image(image_path)

            # Make predictions using the trained model
            prediction = predict_fake_currency(processed_image)

            if prediction == 0:
                result = "Real Currency"
            else:
                result = "Fake Currency"

            return render_template('result.html', result=result,
                                   knn_accuracy=knn_accuracy, cnn_accuracy=max_accuracy, image_path=image_path)


def preprocess_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Resize the image to a fixed size (e.g., 224x224)
    image = cv2.resize(image, (224, 224))

    # Normalize pixel values to the range [0, 1]
    normalized_image = image / 255.0

    # Ensure the image has 3 channels (R, G, B)
    if normalized_image.shape[-1] != 3:
        normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

    return normalized_image


def predict_fake_currency(image):
    # Make the prediction
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Classify as fake (1) if prediction is above a threshold (e.g., 0.5)
    if prediction > 0.5:
        return 1
    else:
        return 0


with open('accuracies.txt', 'r') as file:
    lines = file.readlines()
    knn_accuracy = lines[0].strip()
    max_accuracy = lines[1].strip()

if __name__ == '__main__':
    app.run(debug=True)

# Read the accuracies from the file


# Pass the accuracies to the template
