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

# Path to the video file
video_file_path = "videos/please/PleaseHelpDadTommorrow.mp4"  # Update this with the path to your video file

# Initialize video file capture instead of webcam
cap = cv2.VideoCapture(video_file_path)

# Initialize Mediapipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


class Application(DemoGUI, Pipeline):

    def __init__(self):
        super().__init__()

        self.results = []  # Initialize a list to store the results
        self.is_play_mode = 1  # Set to 'Play mode' by default
        self.notebook.select(1)  # Programmatically select the "Play mode" tab
        self.record_btn_text.set("Record")

        self.hands_detected = False
        self.no_hands_time = None  # Variable to track time when hands are not detected

        self.hands_out_of_frame_duration = .5  # 1 second required for hands to be out of frame
        self.handless_start_time = None  # Timestamp for when hands go out of frame

        self.video_loop()

    def show_frame(self, frame_rgb):
        self.frame_rgb_canvas = frame_rgb
        self.update_canvas()

    def tab_btn_cb(self, event):
        super().tab_btn_cb(event)
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
            res_txt = self.translator_manager.run_knn(feats)
            self.results.append(res_txt)
            self.console_box.delete('1.0', 'end')
            self.console_box.insert('end', f"All results: {self.results}\n")

        else:
            self.knn_records.append(feats)
            self.num_records_text.set(f"num records: {len(self.knn_records)}")

    def save_btn_cb(self):
        super().save_btn_cb()

        gloss_name = self.name_box.get()
        if gloss_name == "":
            logging.error("Empty gloss name.")
            return
        if len(self.knn_records) <= 0:
            logging.error("No knn record found.")
            return

        self.translator_manager.save_knn_database(gloss_name, self.knn_records)
        logging.info("database saved.")
        self.knn_records = []
        self.num_records_text.set("num records: " + str(len(self.knn_records)))
        self.name_box.delete(0, 'end')

    def video_loop(self):
        ret, frame = cap.read()
        if not ret:
            logging.error("Video file finished or camera frame not available.")
            self.close_all()

        frame = utils.crop_utils.crop_square(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        self.hands_detected = results.multi_hand_landmarks is not None

        if self.hands_detected:
            self.handless_start_time = None  # Reset the timer
            cv2.putText(frame_rgb, "Hands Detected", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            if not self.is_recording:
                self.record_btn_cb()

        else:
            cv2.putText(frame_rgb, "No Hands Detected", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            if not self.handless_start_time:
                self.handless_start_time = time.time()

            if time.time() - self.handless_start_time >= self.hands_out_of_frame_duration:
                if self.is_recording:
                    self.record_btn_cb()

        if len(self.results) > 0:
            latest_prediction = self.results[-1]
            cv2.putText(frame_rgb, f"Prediction: {latest_prediction}", (10, 130), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        t1 = time.time()

        self.update(frame_rgb)

        t2 = time.time() - t1
        cv2.putText(frame_rgb, "{:.0f} ms".format(t2 * 1000), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)
        self.show_frame(frame_rgb)

        self.root.after(1, self.video_loop)

    def close_all(self):
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        sys.exit()


if __name__ == "__main__":
    app = Application()
    app.root.mainloop()
