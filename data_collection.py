import logging
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from gui import DemoGUI
from modules import utils
from pipeline import Pipeline

video_folder = 'videos/tomorrow'  # Specify your video folder path here
gloss_name = 'tomorrow'     # Specify your desired gloss name here

class Application(DemoGUI, Pipeline):

    def __init__(self):
        super().__init__()
        # Get list of video files in the folder
        self.video_files = sorted(Path(video_folder).glob('*.mp4'))  # Adjust the extension if needed
        if not self.video_files:
            logging.error(f"No video files found in folder {video_folder}")
            sys.exit()

        self.video_index = 0
        self.cap = None
        self.is_video_playing = False

        # Initialize knn_records list to accumulate features
        self.knn_records = []

        # Set the gloss name in the GUI
        self.name_box.delete(0, 'end')
        self.name_box.insert(0, gloss_name)

        # Start processing videos
        self.process_next_video()

    def process_next_video(self):
        if self.video_index >= len(self.video_files):
            logging.info("All videos processed.")
            # After all videos are processed, save the accumulated data
            self.save_all_records()
            self.close_all()
            return

        video_path = str(self.video_files[self.video_index])
        logging.info(f"Processing video: {video_path}")

        # Initialize video capture for current video
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logging.error(f"Error: Could not open video file {video_path}")
            sys.exit()

        # Start recording
        self.start_recording()

        # Start video loop
        self.is_video_playing = True
        self.video_loop()

    def start_recording(self):
        # Simulate pressing the record button to start recording
        if not self.is_recording:
            self.record_btn_cb()

    def stop_recording(self):
        # Simulate pressing the record button to stop recording
        if self.is_recording:
            self.record_btn_cb()
        # Do not save here; we will save after all videos are processed

    def show_frame(self, frame_rgb):
        self.frame_rgb_canvas = frame_rgb
        self.update_canvas()

    def tab_btn_cb(self, event):
        super().tab_btn_cb(event)
        # Check database before changing from record mode to play mode.
        if self.is_play_mode:
            ret = self.translator_manager.load_knn_database()
            if not ret:
                logging.error("KNN Sample is missing. Please record some samples before starting play mode.")
                self.notebook.select(0)

    def record_btn_cb(self):
        super().record_btn_cb()
        # When starting recording, simply return
        if self.is_recording:
            return

        # When stopping recording, process the recorded data
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

        # Accumulate feats into knn_records
        self.knn_records.append(feats)
        self.num_records_text.set(f"num records: {len(self.knn_records)}")

    def save_all_records(self):
        # Read gloss name from the text entry
        gloss_name_input = self.name_box.get()

        if gloss_name_input == "":
            logging.error("Empty gloss name.")
            return
        if len(self.knn_records) <= 0:
            logging.error("No KNN records to save.")
            return

        # Save the accumulated knn_records under the specified gloss name
        self.translator_manager.save_knn_database(gloss_name_input, self.knn_records)

        logging.info(f"Database saved with gloss name '{gloss_name_input}'.")

        # Clear knn_records and reset GUI elements
        self.knn_records = []
        self.num_records_text.set("num records: 0")
        self.name_box.delete(0, 'end')

    def video_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.info("Video ended.")
            self.stop_recording()
            self.is_video_playing = False
            self.video_index += 1
            self.process_next_video()
            return

        frame = utils.crop_utils.crop_square(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t1 = time.time()

        # Update pipeline with the current frame
        self.update(frame_rgb)

        t2 = time.time() - t1
        cv2.putText(frame_rgb, "{:.0f} ms".format(t2 * 1000), (10, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)
        self.show_frame(frame_rgb)

        # Continue looping
        self.root.after(1, self.video_loop)

    def close_all(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()


if __name__ == "__main__":
    app = Application()
    app.root.mainloop()
