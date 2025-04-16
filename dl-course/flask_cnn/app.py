import os
from flask import Flask, request, render_template, jsonify
from model import simplecnn  # Assuming 'model.py' in the same directory
import numpy as np
from PIL import Image
import io
import torch
from torchvision import transforms

app = Flask(__name__)
classes = ['猫', '狗']

# Global variable to store the loaded model
model = None

def load_model():
    """Loads the trained model."""
    global model  # Access the global model variable
    if model is None:
        model = simplecnn(len(classes)) # Corrected here
        model_path = './model_pt/model.pt'
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Ensure CPU if no GPU
            model.load_state_dict(checkpoint)
            model.eval()  # Set to evaluation mode
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None
    return model



def predict_image(img_file):
    """Preprocesses the image and returns the prediction."""
    try:
        img = Image.open(io.BytesIO(img_file.read())).convert('RGB')
    except Exception as e:
        print(f"Error opening or converting the image: {e}")
        return None

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Corrected Normalization value
    ])
    try:
       img_tensor = transform(img).unsqueeze(0)
    except Exception as e:
        print(f"Error transforming image : {e}")
        return None

    model = load_model()

    if model is None:
      return None

    with torch.no_grad():  # Disable gradient calculations
        try:
            output = model(img_tensor)
            _, pred = torch.max(output, 1)  # Use torch.max for getting prediction
            return pred.item()  # Get the predicted class as an integer
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='没有选择图片文件')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='没有选择图片文件')

        if file:
            pred = predict_image(file)
            if pred is not None:
              result = classes[pred]
              result = f'这是一只 {result}'
              return render_template('index.html', result=result)
            else:
              return render_template('index.html', message='Error during prediction')
    return render_template('index.html')

#Added an endpoint to get result through json
@app.route('/predict',methods=['POST'])
def predict_api():
    if 'file' not in request.files:
         return jsonify({'error':'No file uploaded'}),400
    file = request.files['file']
    if file.filename =='':
         return jsonify({'error': 'No file uploaded'}),400
    if file:
       pred =predict_image(file)
       if pred is not None:
           result = classes[pred]
           return jsonify({'result':result}),200
       else:
           return jsonify({'error':'Error during prediction'}),500

    return jsonify({'error': 'Unknown Error'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)