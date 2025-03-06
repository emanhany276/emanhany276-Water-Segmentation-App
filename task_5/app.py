from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import tensorflow as tf
import rasterio
import os
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("water_segmentation_model.h5", compile=False)

# Min-Max normalization function for input images
def normalize_image(image):
    min_val = np.min(image, axis=(0, 1), keepdims=True)
    max_val = np.max(image, axis=(0, 1), keepdims=True)
    normalized_image = (image - min_val) / (max_val - min_val + 1e-7)
    return normalized_image

# Load and preprocess the TIFF image
def load_and_preprocess_image(file):
    with rasterio.open(file) as src:
        image = src.read().astype(np.float32)
        image = np.transpose(image, (1, 2, 0))  # Convert to (128, 128, 12)

    # Normalize image
    normalized_image = normalize_image(image)
    return normalized_image

# Home route to render the HTML UI
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Preprocess and predict
        image = load_and_preprocess_image(file)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image)[0, ..., 0]

        # Convert prediction to binary mask
        binary_mask = (prediction > 0.5).astype(np.uint8) * 255

        # Save mask as PNG in memory
        mask_img = Image.fromarray(binary_mask)
        byte_io = io.BytesIO()
        mask_img.save(byte_io, 'PNG')
        byte_io.seek(0)

        return send_file(byte_io, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
