from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from io import BytesIO
from sleeve_classifier import YourModelClass  # Import your model from classifier.py

app = Flask(__name__)

# Load the trained model
model = YourModelClass()  # Instantiate your model from classifier.py

# Define image preprocessing function
def preprocess_image(image):
    img = Image.open(BytesIO(image))
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return img

# Define root route to serve HTML form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        img_bytes = file.read()
        img = preprocess_image(img_bytes)
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img)
        print(prediction)
        # Map class indices to sleeve types
        sleeve_index = np.argmax(prediction[0])
        
        # Determine gender and sleeve type based on indices
        sleeve_type = 'full sleeve' if sleeve_index == 0 else 'half sleeve'
        
        # Return prediction
        return jsonify({'sleeve type': sleeve_type})

if __name__ == '__main__':
    app.run(debug=False)
