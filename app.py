from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model = load_model('model.h5')

class_labels = ['Cattleya', 'Dendrobium', 'Oncidium', 'Phalaenopsis', 'Vanda']

def predict_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    predictions = model.predict(image)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is submitted
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Predict the image
            predicted_class = predict_image(file_path)
            
            # Render result
            return render_template('index.html', filename=filename, prediction=predicted_class)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return url_for('static', filename='uploads/' + filename)

if __name__ == '__main__':
    app.run(debug=True)