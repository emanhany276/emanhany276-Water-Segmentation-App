# **Water Segmentation Web Application**

This repository contains a web application for water segmentation using multispectral and optical data. It allows users to upload TIFF images, preprocess them, and generate segmented water masks using a trained deep learning model. The web interface is built with Flask and TailwindCSS, and the backend uses TensorFlow for prediction.

---



## **Features**
- Upload multispectral TIFF images for segmentation.
- Preprocess images with min-max normalization.
- Generate binary water masks using a trained TensorFlow model.
- Display the segmented mask in the web UI.
- Error handling for file upload and prediction failures.

---

## **Requirements**
- Python 3.8+
- Flask
- TensorFlow
- NumPy
- rasterio
- Pillow (PIL)

Install the requirements using:
```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
Flask
tensorflow
numpy
rasterio
Pillow
```

---



## **Usage**
1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Open in browser:**
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. **Upload a TIFF file:**
   - Click "Choose File" and select a `.tif` or `.tiff` image.
   - Click "Upload" to generate the water mask.
   - The segmented mask will be displayed below the upload button.

---

## **Project Structure**
```
water-segmentation/
├── app.py               # Flask application code
├── water_segmentation_model.h5 # Trained model file (not included in the repo)
├── templates/
│   └── index.html       # HTML for web UI

```

---

## **Model**
- The model is a TensorFlow-based deep learning model trained to perform water segmentation on multispectral data.
- It outputs a probability mask where each pixel indicates the probability of being water.
- The binary mask is created using a threshold of 0.5.

---

## **Normalization**
- The model expects normalized input:
  - Min-Max normalization is applied to each band of the image.
  - Formula used:  
    \[
    \text{normalized\_image} = \frac{\text{image} - \min}{\max - \min + 1e-7}
    \]
- Ensures values are scaled between 0 and 1.

---

## **Error Handling**
- **File Upload:**
   - Returns a 400 error if no file is uploaded or if the file type is unsupported.
- **Prediction:**
   - Returns a 500 error if an exception occurs during prediction.

---



**Enjoy using the Water Segmentation Web App! 🌊**
![ws](https://github.com/user-attachments/assets/2ded7151-c98d-4a85-99f2-6718950329bc)

