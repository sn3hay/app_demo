import cv2
import numpy as np
import base64
from PIL import Image
import io
import mediapipe as mp
import pickle
import random
import os

# random_char = 'A'
# random_char_index = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# inputs = pickle.load(open('cnn_model.p','rb'))

# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the cnn_model.p file
model_path = os.path.join(dir_path, 'cnn_model.p')

# Load the model
inputs = pickle.load(open(model_path, 'rb'))
model = inputs['model']
label_dict = inputs['label_dict']

def get_random_char():
    global random_char
    global random_char_index

    random_char = random.choice(list(label_dict.values()))
    
    random_char_index = list(label_dict.values()).index(random_char)
    print('In processing: ',random_char)
    return random_char, random_char_index

def decode_image(data):
    try:
        header, encoded = data.split(',', 1)
        binary_data = base64.b64decode(encoded)
        img_array = np.frombuffer(binary_data, dtype=np.uint8)
        if img_array.size == 0:
            print("Received empty image buffer")
            return None
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode image")
        return img
    except Exception as e:
        print(f"Failed to process image data: {str(e)}")
        return None


def process_image(img):
    """Process image to extract hand landmarks and calculate bounding box."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    bounding_box = {}
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_coords = []
            y_coords = []
            for landmark in hand_landmarks.landmark:
                x, y = landmark.x * img.shape[1], landmark.y * img.shape[0]
                data_aux.extend([x, y])
                x_coords.append(x)
                y_coords.append(y)
            if len(data_aux) == 42:
                hand_data = np.array(data_aux, dtype=np.float32).reshape(1, -1)
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bounding_box = {'xmin': x_min, 'xmax': x_max, 'ymin': y_min, 'ymax': y_max}
                return hand_data, bounding_box
    return None, bounding_box


def get_prediction(hand_data):
    """Predict the ASL letter from the processed hand landmarks."""
    if hand_data is not None:
        prediction = model.predict(hand_data)
        predicted_letter = label_dict[np.argmax(prediction)]
        return predicted_letter
    return 'No hand detected'

def get_prediction_prob(hand_data):
    """Predict the ASL letter from the processed hand landmarks."""
    print('Inside get_prediction_prob')
    if hand_data is not None:
        prediction = model.predict(hand_data)
        prediction_prob = prediction[0][random_char_index]
        print('Prediction prob =>',prediction_prob)
        prediction_prob = prediction_prob.item() 
        return prediction_prob
    return 'No hand detected'