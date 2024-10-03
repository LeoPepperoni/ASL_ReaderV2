import sys

import cv2
import numpy as np
import mediapipe as mp  # Importing Mediapipe

from modules import utils
from modules.translator import TranslatorManager
from pipeline import Pipeline
from flask import Flask, jsonify, request
import threading

cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


class Application(Pipeline):

    def __init__(self):
        super().__init__()

        self.results = []  # Initialize a list to store the results
        self.app = Flask(__name__)
        self.pose_history = []
        self.face_history = []
        self.lh_history = []
        self.rh_history = []
        self.translator_manager = TranslatorManager()

            # Flag to check if hands are detected
        self.hands_detected = False

        self.video_loop()

        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.add_url_rule('/results', 'get_results',
                              self.get_results)  # Calling this route will server results to the front-end
        # Define the route to accept video data
        self.app.add_url_rule('/upload_video', 'upload_video', self.upload_video, methods=['POST'])
        # Start the Flask app in a separate thread
        threading.Thread(target=self.run_flask_app, daemon=True).start()

    def run_flask_app(self):
        self.app.run(host='0.0.0.0', port=8080)

    def upload_video(self):

        if 'video' not in request.files:
            return jsonify({"error": "No video part in the request"}), 400

        video_file = request.files['video']

        if video_file.filename == '':
            return jsonify({"error": "No video selected"}), 400

        frame = utils.crop_utils.crop_square(video_file)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        vid_res = {
            "pose_frames": np.stack(self.pose_history),
            "face_frames": np.stack(self.face_history),
            "lh_frames": np.stack(self.lh_history),
            "rh_frames": np.stack(self.rh_history),
            "n_frames": len(self.pose_history)
        }
        feats = self.translator_manager.get_feats(vid_res)
        self.reset_pipeline()
        threading.Thread(target=self.run_prediction, args=(feats,)).start()

        # Return a response
        return jsonify({"message": f"Video {video_file.filename} uploaded successfully"}), 200

    def get_results(self):
        return jsonify(self.results)  # Return results as JSON

    def run_prediction(self, feats):
        res_txt = self.translator_manager.run_knn(feats)
        self.results.append(res_txt)  # Store result in the results list
        self.console_box.insert('end', f"All results: {self.results}\n")

    def close_all(self):
        cap.release()
        hands.close()  # Close Mediapipe hand detection
        cv2.destroyAllWindows()
        sys.exit()


if __name__ == "__main__":
    app = Application()
