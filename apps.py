from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
from PIL import Image
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for your app if needed.

# Load the CNN model
modelxception94persen = load_model("Rafi-garbageXception-94.14.h5")
# modelxception94persen = load_model("Rafi-garbageXception-93.72.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("cnn.html")

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classifications.html")

@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('file')
    filename = "temp_image.png"
    errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors["message"] = 'File type of {} is not allowed'.format(file.filename)

    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp
    img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Convert the image to RGB
    img = Image.open(img_url).convert('RGB')
    now = datetime.now()
    predict_image_path = 'static/uploads/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    image_predict = predict_image_path
    img.convert('RGB').save(image_predict, format="png")
    img.close()

    # Prepare the image for prediction
    img = image.load_img(predict_image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize the image

    # Predict using the CNN model
    prediction_array_xception94persen = modelxception94persen.predict(x)
    predicted_class_index = np.argmax(prediction_array_xception94persen)
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
    predicted_class_name = class_names[predicted_class_index]
    predicted_confidence = 100 * prediction_array_xception94persen[0][predicted_class_index]

    return render_template("classifications.html", img_path=predict_image_path,
                           predictionxception94persen=predicted_class_name,
                           confidenceexception94persen='{:2.2f}%'.format(predicted_confidence))

if __name__ == '__main__':
    app.run(debug=True)
