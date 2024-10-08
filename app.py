import sys

import cv2
import numpy as np
import mediapipe as mp  # Importing Mediapipe

from modules import utils
from modules.translator import TranslatorManager
from pipeline import Pipeline
from flask import Flask, jsonify, request
import threading
import random
import os

cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

fake_words = [
    "apple", "banana", "cherry", "date", "elderberry",
    "fig", "grape", "honeydew", "kiwi", "lemon",
    "mango", "nectarine", "orange", "papaya", "quince",
    "raspberry", "strawberry", "tangerine", "ugli", "vanilla",
    "watermelon", "xigua", "yellowberry", "zucchini"
]

# Initialize Flask app
app = Flask(__name__)

pose_history = []
face_history = []
lh_history = []
rh_history = []
translator_manager = TranslatorManager()
hands_detected = False

# Sample results
# results = ["apple", "banana", "cherry", "date", "elderberry", "fig"]
results = []
UPLOAD_FOLDER = "videos/sample_video.mp4"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/results', methods=["GET"])
def get_results():
    if results:
        return jsonify({"response": results})  # Remove the space in "response"
    else:
        return jsonify({"response": "No results to show"})  # Handle empty results


@app.route("/upload", methods=["POST"])# save the video from the fornt end and then send it to model
def upload_video(self):
    if 'video' not in request.files:
        return jsonify({"error": "No video part in the request"}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({"error": "No video selected"}), 400

    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Save the video file to the specified path
    video_file_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_file_path)

    # Return a response
    return jsonify({"message": f"Video {video_file.filename} uploaded successfully"}), 200


# class Application(Pipeline):

# def __init__(self):
#     super().__init__()


# self.app.add_url_rule('/results', 'get_results', self.get_results)  # Calling this route will server results to the front-end
# # Define the route to accept video data
# self.app.add_url_rule('/upload_video', 'upload_video', self.upload_video, methods=['POST'])

def run_prediction(self, feats):
    res_txt = self.translator_manager.run_knn(feats)
    self.results.append(res_txt)  # Store result in the results list


def close_all(self):
    cap.release()
    hands.close()  # Close Mediapipe hand detection
    cv2.destroyAllWindows()
    sys.exit()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)