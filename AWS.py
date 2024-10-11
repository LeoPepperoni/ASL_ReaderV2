import logging
import sys
import time
from pathlib import Path

import threading
import cv2
import numpy as np
import mediapipe as mp  # Importing Mediapipe
import os
from flask import Flask, jsonify, request
from gui import DemoGUI
from modules import utils
from pipeline import Pipeline

# Path to the video file
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Mediapipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


class Application(Pipeline):

    def __init__(self):
        super().__init__()

        self.hands_out_of_frame_duration = 1  # Time threshold for hands being out of frame
        self.handless_start_time = None  # Timestamp for when hands go out of frame

        self.results = []  # Initialize a list to store the results
        self.hands_detected = False  # Flag to check if hands are detected

        # Initialize Flask & Define the routes
        self.app = Flask(__name__)
        self.app.add_url_rule('/results', 'get_results', self.get_results)  # Route to get results
        self.app.add_url_rule('/upload_video', 'upload_video', self.upload_video, methods=['POST'])
        self.app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        self.app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

        self.app.config['uploads'] = "uploads"
        os.makedirs(self.app.config['uploads'], exist_ok=True)

    def run_flask_app(self):
        self.app.run(debug=True, host='0.0.0.0', port=8080, use_reloader=False)

    def get_results(self):
        if self.results:
            return jsonify({'response': self.results})  # Return results as JSON
        else:
            return jsonify({'response': 'no results to show'})  # Return results as JSON

    def upload_video(self):
        data = self.translator_manager.load_knn_database()
        video_file_path = os.path.join('uploads', 'Pleasehelpdad.mp4')
        cap = cv2.VideoCapture(video_file_path)  # reads in file from specified path

        if not cap.isOpened():
            print(f"Error opening video file")
            return
        test = 0
        while True:  # will run until a condition breaks it
            ret, frame = cap.read()

            if not ret:  # condition to break out of the loop, if a frame is not returned
                break

            frame = utils.crop_utils.crop_square(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame.any() and test == 0:
                print('received frame')
                test += 1

            # Mediapipe Hand Detection
            results = hands.process(frame_rgb)

            # Check if hands are detected
            self.hands_detected = results.multi_hand_landmarks is not None

            if self.hands_detected:
                self.handless_start_time = None  # Reset the timer if hands are detected
            else:
                # If hands are not detected, start or continue the timer
                if self.handless_start_time is None:
                    self.handless_start_time = time.time()  # Start the timer
                elif time.time() - self.handless_start_time >= self.hands_out_of_frame_duration:
                    # If hands have been out of frame for 0.75 seconds, make prediction
                    if len(self.pose_history) >= 16:  # Ensure sufficient history
                        vid_res = {
                            "pose_frames": np.stack(self.pose_history),
                            "face_frames": np.stack(self.face_history),
                            "lh_frames": np.stack(self.lh_history),
                            "rh_frames": np.stack(self.rh_history),
                            "n_frames": len(self.pose_history)
                        }
                        feats = self.translator_manager.get_feats(vid_res)

                        if data:
                            res_txt = self.translator_manager.run_knn(feats)
                            self.results.append(res_txt)
                        self.reset_pipeline()

            self.update(frame_rgb)
        cap.release()

        return self.results

if __name__ == "__main__":
    app = Application()
    response = app.upload_video()
    print(response)
