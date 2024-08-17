from flask import Flask, request, render_template
import numpy as np
import cv2
import json
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('models/crop_disease_model_resnet50.h5')
IMAGE_SIZE = (224, 224)

# Load class labels from JSON file
with open('models/class_labels.json', 'r') as f:
    class_indices = json.load(f)
class_labels = list(class_indices.keys())

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict the disease
def predict_disease(image_path):
    try:
        img = preprocess_image(image_path)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_disease = class_labels[predicted_class[0]]

        print(f"Predicted Disease: {predicted_disease}")  # Debugging print statement
        
        # Get precautions for the predicted disease
        precautions = disease_precautions.get(predicted_disease, ["No specific precautions available."])
        
        print(f"Precautions: {precautions}")  # Debugging print statement

        return predicted_disease, precautions
    except Exception as e:
        return f"Error in prediction: {str(e)}", []


# Dictionary mapping diseases to precautions
disease_precautions = {
    "Tomato_Leaf_Mold": [
        "Use disease-free seeds and transplants.",
        "Avoid overhead irrigation to reduce leaf wetness.",
        "Apply copper-based bactericides early in the growing season.",
        "Practice crop rotation to prevent recurrence."
    ],
    "Potato___Early_blight": [
        "Plant resistant varieties whenever possible.",
        "Ensure proper spacing to improve air circulation.",
        "Remove and destroy affected leaves.",
        "Apply fungicides if necessary, following local guidelines."
    ],
    "Pepper__bell___Bacterial_spot": [
        "Use disease-free seeds and resistant varieties.",
        "Avoid working with plants when they are wet to reduce spread.",
        "Remove and destroy infected plants.",
        "Implement crop rotation and avoid planting peppers in the same area consecutively."
    ],
    # Add other diseases and their precautions here if needed
}

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the file to a temporary location
            filepath = os.path.join('static/uploads', file.filename)
            file.save(filepath)
            
            # Predict the disease and get precautions
            prediction, precautions = predict_disease(filepath)
            
            # Redirect to the result page with the prediction and precautions
            return render_template('result.html', prediction=prediction, precautions=precautions, image_path=filepath)

    
    return render_template('index.html')

# Route for the result page
@app.route('/result')
def result():
    return render_template('result.html')

# Start the app
if __name__ == '__main__':
    app.run(debug=True)
