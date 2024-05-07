from flask import Flask, render_template, request, jsonify, Response
import cv2
import base64
import numpy as np
from processing import process_image, decode_image, get_prediction, get_random_char, get_prediction_prob
import awsgi 

app = Flask(__name__)

# def lambda_handler(event, context):
#     return awsgi.response(app, event, context)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/english_to_asl')
def english_to_asl():
    return render_template('english_to_asl.html')



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image']
    img = decode_image(img_data)
    if img is not None:
        hand_landmarks, bbox = process_image(img)
        if hand_landmarks is not None:
            prediction = get_prediction(hand_landmarks)
            return jsonify({'char': prediction, 'bbox': bbox})
        else:
            return jsonify({'char': 'No hand detected', 'bbox': {}})
    else:
        return jsonify({'error': 'Invalid image data', 'bbox': {}})

@app.route('/predict_prob', methods=['POST'])
def predict_prob():
    data = request.get_json()
    img_data = data['image']
    img = decode_image(img_data)
    if img is not None:
        hand_landmarks, bbox = process_image(img)
        if hand_landmarks is not None:
            probability = get_prediction_prob(hand_landmarks)
            bbox = {key: float(val) for key, val in bbox.items()}
            return jsonify({'probability': probability, 'bbox': bbox})
        else:
            return jsonify({'probability': 'No hand detected', 'bbox': {}})
    else:
        return jsonify({'error': 'Invalid image data', 'bbox': {}})


@app.route('/get_random_letter', methods=['POST'])
def get_random_letter():
    data = request.get_json()
    random_char, random_char_index = get_random_char()
    print('In app: ',random_char)
    return jsonify({'random_char': random_char})

if __name__ == '__main__':
    app.run(debug=True)