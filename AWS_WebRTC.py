import logging
import sys
import time
from pathlib import Path
import threading
import cv2
import numpy as np
import mediapipe as mp  # Importing Mediapipe
import os
from flask import Flask, jsonify, request, Response
from gui import DemoGUI
import socketio  # Importing Socket.IO
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from modules import utils
from pipeline import Pipeline
import ssl  # Added for HTTPS support

# Initialize Mediapipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi')
app = Flask(__name__)
sio_app = socketio.WSGIApp(sio, app)

# Initialize WebRTC components
peer_connection = None
media_relay = MediaRelay()
ice_servers = [
    {
        'urls': ['stun:stun.l.google.com:19302', 'stun:stun1.l.google.com:19302']
    }
]

class VideoProcessor(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track  # MediaStreamTrack (from WebRTC)
        self.results = []  # Store results from video processing
        self.hands_detected = False
        self.hands_out_of_frame_duration = 0.75
        self.handless_start_time = None
        self.pose_history, self.face_history, self.lh_history, self.rh_history = [], [], [], []

    async def recv(self):
        frame = await self.track.recv()  # Receive video frame from WebRTC stream

        # Convert the frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mediapipe Hand Detection
        results = hands.process(frame_rgb)

        # Check if hands are detected
        self.hands_detected = results.multi_hand_landmarks is not None

        if self.hands_detected:
            self.handless_start_time = None  # Reset the timer if hands are detected
            self.update(frame_rgb)
        else:
            # If hands are not detected, start or continue the timer
            if self.handless_start_time is None:
                self.handless_start_time = time.time()  # Start the timer
            elif time.time() - self.handless_start_time >= self.hands_out_of_frame_duration:
                # If hands have been out of frame for 0.75 seconds, make a prediction
                if len(self.pose_history) >= 16:  # Ensure sufficient history
                    vid_res = {
                        "pose_frames": np.stack(self.pose_history),
                        "face_frames": np.stack(self.face_history),
                        "lh_frames": np.stack(self.lh_history),
                        "rh_frames": np.stack(self.rh_history),
                        "n_frames": len(self.pose_history)
                    }

                    # Call the translator manager for predictions
                    feats = self.translator_manager.get_feats(vid_res)
                    self.reset_pipeline()

                    # Retrieve KNN results and append to results
                    data = self.translator_manager.load_knn_database()
                    if data:
                        res_txt = self.translator_manager.run_knn(feats)
                        self.results.append(res_txt)

        return frame  # Return the original frame for WebRTC to process



class Application(Pipeline):

    def __init__(self):
        super().__init__()
        # Flask App Setup
        self.app = Flask(__name__)
        self.app.add_url_rule('/results', 'get_results', self.get_results)  # Route to get results

    def run_flask_app(self):
        # Set the path to the SSL certificate and key
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile='path/to/certfile.pem', keyfile='path/to/keyfile.pem')

        # Run the Flask app with HTTPS and WebSocket support
        from werkzeug.serving import make_server
        server = make_server('0.0.0.0', 8080, sio_app, ssl_context=ssl_context)
        server.serve_forever()

    def get_results(self):
        if self.results:
            return jsonify({'response': self.results})  # Return results as JSON
        else:
            return jsonify({'response': 'no results to show'})  # Return results as JSON

# SOCKET.IO EVENTS for WebRTC
@sio.event
async def connect(sid, environ, auth):
    print('Client connected:', sid)
    if auth['password'] != 'x':
        return False

@sio.event
async def newOffer(sid, offer):
    global peer_connection
    peer_connection = await create_peer_connection()
    await peer_connection.setRemoteDescription(RTCSessionDescription(sdp=offer['sdp'], type=offer['type']))
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)
    await sio.emit('answerResponse', {'answer': answer})

@sio.event
async def sendIceCandidateToSignalingServer(sid, data):
    candidate = data.get('iceCandidate')
    if candidate:
        await peer_connection.addIceCandidate(candidate)

@sio.event
async def hangup(sid):
    global peer_connection
    if peer_connection:
        await peer_connection.close()
        peer_connection = None
    print("Call ended")

async def create_peer_connection():
    global peer_connection
    peer_connection = RTCPeerConnection(configuration={"iceServers": ice_servers})

    @peer_connection.on("icecandidate")
    async def on_icecandidate(ice_candidate):
        if ice_candidate:
            await sio.emit('sendIceCandidateToSignalingServer', {
                'iceCandidate': ice_candidate
            })

    @peer_connection.on("track")
    def on_track(track):
        print(f"Track {track.kind} received")
        if track.kind == "video":
            video_processor = VideoProcessor(track)
            peer_connection.addTrack(video_processor)  # Process video track in real-time

    return peer_connection


if __name__ == "__main__":
    app = Application()
    app.run_flask_app()
