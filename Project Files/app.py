from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# ✅ Load trained model
MODEL_PATH = 'healthy_vs_rotten.h5'
model = load_model(MODEL_PATH)

# ✅ Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Class names from your training
classes = [
    'Apple__Healthy', 'Apple__Rotten', 'Banana__Healthy', 'Banana__Rotten',
    'Bellpepper__Healthy', 'Bellpepper__Rotten', 'Carrot__Healthy', 'Carrot__Rotten',
    'Cucumber__Healthy', 'Cucumber__Rotten', 'Grape__Healthy', 'Grape__Rotten',
    'Guava__Healthy', 'Guava__Rotten', 'Jujube__Healthy', 'Jujube__Rotten',
    'Mango__Healthy', 'Mango__Rotten', 'Orange__Healthy', 'Orange__Rotten',
    'Pomegranate__Healthy', 'Pomegranate__Rotten', 'Potato__Healthy', 'Potato__Rotten',
    'Strawberry__Healthy', 'Strawberry__Rotten', 'Tomato__Healthy', 'Tomato__Rotten'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # ✅ Show upload form
        return render_template('predict.html')

    # ✅ Handle POST (image upload and prediction)
    if 'image' not in request.files:
        return 'No image uploaded.'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file.'

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # ✅ Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ Run prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)

        # ✅ Map index to class name
        if predicted_class_index < len(classes):
            predicted_class = classes[predicted_class_index]
        else:
            predicted_class = "Unknown Class"

        return render_template('output.html', prediction_text=predicted_class, image_path='/' + filepath)

if __name__ == "__main__":
    app.run(debug=True)
