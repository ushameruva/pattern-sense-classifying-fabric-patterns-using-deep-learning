from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model('model_cnn.h5')  # Make sure this file exists

# Your class labels
labels = ['animal', 'cartoon', 'floral', 'geometry', 'ikat', 'plain', 
          'polka dot', 'squares', 'stripes', 'tribal']

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# === ROUTES ===

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Contact route
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Classify image route
@app.route('/classify')
def classify():
    return render_template('predict.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded.', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file.', 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = prediction[predicted_index]

    return render_template('predictionpage.html',
                           label=predicted_label,
                           confidence=f"{confidence*100:.2f}",
                           image_file=filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
