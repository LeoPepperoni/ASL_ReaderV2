import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp  # Importing Mediapipe

from gui import DemoGUI
from modules import utils
from pipeline import Pipeline
from flask import Flask, jsonify, request
import threading

cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


class Application(DemoGUI, Pipeline):

    def __init__(self):
        super().__init__()

        self.results = []  # Initialize a list to store the results

        # Set to Play Mode initially
        self.is_play_mode = 1  # Set to 'Play mode' by default
        self.notebook.select(1)  # Programmatically select the "Play mode" tab

        # Update record button text to reflect play mode
        self.record_btn_text.set("Record")

        # Flag to check if hands are detected
        self.hands_detected = False

        self.video_loop()

        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.add_url_rule('/results', 'get_results',self.get_results)  # Calling this route will server results to the front-end
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

        # Read the image as an OpenCV format
        frame_bytes = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

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

    def show_frame(self, frame_rgb):
            self.frame_rgb_canvas = frame_rgb
            self.update_canvas()

    def tab_btn_cb(self, event):
        super().tab_btn_cb(event)
        # check database before change from record mode to play mode.
        if self.is_play_mode:
            ret = self.translator_manager.load_knn_database()
            if not ret:
                logging.error("KNN Sample is missing. Please record some samples before starting play mode.")
                self.notebook.select(0)


    def record_btn_cb(self):
        super().record_btn_cb()
        if self.is_recording:
            return

        if len(self.pose_history) < 16:
            logging.warning("Video too short.")
            self.reset_pipeline()
            return

        vid_res = {
            "pose_frames": np.stack(self.pose_history),
            "face_frames": np.stack(self.face_history),
            "lh_frames": np.stack(self.lh_history),
            "rh_frames": np.stack(self.rh_history),
            "n_frames": len(self.pose_history)
        }
        feats = self.translator_manager.get_feats(vid_res)
        self.reset_pipeline()

        # Play mode: run translator.
        if self.is_play_mode:
            self.console_box.delete('1.0', 'end')
            threading.Thread(target=self.run_prediction, args=(feats,)).start()

            # Update the GUI in a thread-safe manner
        else:
            self.knn_records.append(feats)
            self.num_records_text.set(f"num records: {len(self.knn_records)}")


    def run_prediction(self, feats):
        res_txt = self.translator_manager.run_knn(feats)
        self.results.append(res_txt)  # Store result in the results list
        self.console_box.insert('end', f"All results: {self.results}\n")


    def save_btn_cb(self):
        super().save_btn_cb()

        # Read texbox entry, use as folder name.
        gloss_name = self.name_box.get()

        if gloss_name == "":
            logging.error("Empty gloss name.")
            return
        if len(self.knn_records) <= 0:
            logging.error("No knn record found.")
            return

        self.translator_manager.save_knn_database(gloss_name, self.knn_records)

        logging.info("database saved.")
        # clear.
        self.knn_records = []
        self.num_records_text.set("num records: " + str(len(self.knn_records)))
        self.name_box.delete(0, 'end')

    def video_loop(self):
        ret, frame = cap.read()
        if not ret:
            logging.error("Camera frame not available.")
            self.close_all()

        frame = utils.crop_utils.crop_square(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe Hand Detection
        results = hands.process(frame_rgb)

        # Check if hands are detected
        self.hands_detected = results.multi_hand_landmarks is not None

        if self.hands_detected:
            cv2.putText(frame_rgb, "Hands Detected", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            if self.is_recording == False: self.record_btn_cb()

        else:
            cv2.putText(frame_rgb, "No Hands Detected", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            if self.is_recording == True: self.record_btn_cb()

        t1 = time.time()

        self.update(frame_rgb)

        t2 = time.time() - t1
        cv2.putText(frame_rgb, "{:.0f} ms".format(t2 * 1000), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)
        self.show_frame(frame_rgb)

        self.root.after(1, self.video_loop)

    # def update_console(self):
    #     # Display all results in the console
    #     self.console_box.delete('1.0', 'end')
    #     self.console_box.insert('end', f"All results: {self.results}\n")  # Show all results

    def close_all(self):
        cap.release()
        hands.close()  # Close Mediapipe hand detection
        cv2.destroyAllWindows()
        sys.exit()


if __name__ == "__main__":
    app = Application()
    app.root.mainloop()
