import cv2  # For Computer Vision
try:
    import tensorflow as tf  # For AI Models
except ImportError as e:
    print("Error: TensorFlow could not be imported. Please ensure it is installed.")
    raise e
from transformers import pipeline  # For NLP
import speech_recognition as sr  # For real-time audio transcription
import numpy as np  # For array manipulation
from flask import Flask, request, jsonify  # For Web API
import os
import json
import logging  # For logging
from moviepy.editor import VideoFileClip  # For audio extraction
from pytube import YouTube  # For YouTube video downloading

# Initialize logging
logging.basicConfig(filename='filter.log', level=logging.INFO)

# Flask app for backend
app = Flask(__name__)

# ==================================
# AI Modules
# ==================================

# Load YOLO model for objectionable content detection
def load_yolo_model():
    """
    Load the pre-trained YOLO model for object detection.
    :return: Loaded YOLO model and class labels.
    """
    weights_path = "yolov3.weights"  # Path to YOLO weights
    config_path = "yolov3.cfg"       # Path to YOLO config
    class_labels_path = "coco.names" # Path to class labels

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    with open(class_labels_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, classes

def detect_objectionable_content_yolo(frame, net, classes):
    """
    Detect objectionable content in a video frame using YOLO.
    :param frame: Single frame from the video stream.
    :param net: YOLO model.
    :param classes: Class labels for YOLO.
    :return: True if objectionable content is detected, False otherwise.
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    objectionable = False
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["nude", "explicit"]:
                objectionable = True
                break
    return objectionable

def transcribe_and_analyze_audio(audio):
    """
    Transcribe audio and analyze for objectionable language.
    :param audio: Audio clip or stream.
    :return: True if objectionable language is detected, False otherwise.
    """
    recognizer = sr.Recognizer()
    objectionable = False
    try:
        transcription = recognizer.recognize_google(audio)
        nlp_model = pipeline("text-classification", model="distilbert-base-uncased")
        analysis = nlp_model(transcription)
        # Example: Check for categories like 'offensive', 'profanity', etc.
        objectionable = any(item['label'] in ['HATE', 'PROFANITY'] for item in analysis)
    except sr.RequestError:
        print("Error: Could not connect to speech recognition service")
    except sr.UnknownValueError:
        print("Error: Could not understand audio")
    return objectionable

def apply_filters(frame, preferences, net, classes, cap, current_frame):
    """
    Apply filters to video frames based on user preferences.
    :param frame: Video frame to analyze and modify.
    :param preferences: User preferences for filtering.
    :param net: YOLO model for detection.
    :param classes: Class labels for YOLO.
    :param cap: Video capture object.
    :param current_frame: Current frame number.
    :return: Processed frame.
    """
    # Video filtering
    if preferences.get('blur') and detect_objectionable_content_yolo(frame, net, classes):
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        logging.info(f"Blurred frame at timestamp {current_frame / cap.get(cv2.CAP_PROP_FPS):.2f} seconds.")

    # Scene skipping
    if preferences.get('skip') and detect_objectionable_content_yolo(frame, net, classes):
        SKIP_FRAMES = 150  # Skip ahead 150 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + SKIP_FRAMES)
        logging.info(f"Skipped frames at timestamp {current_frame / cap.get(cv2.CAP_PROP_FPS):.2f} seconds.")

    return frame

# ==================================
# YouTube Video Download
# ==================================

def download_youtube_video(url, output_path="downloads"):
    """
    Download a YouTube video and save it locally.
    :param url: URL of the YouTube video.
    :param output_path: Directory to save the video.
    :return: Path to the downloaded video.
    """
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    video_path = stream.download(output_path)
    print(f"Downloaded video to {video_path}")
    return video_path

# ==================================
# Audio Extraction
# ==================================

def extract_audio(video_path, audio_path="output_audio.wav"):
    """
    Extract audio from a video file.
    :param video_path: Path to the video file.
    :param audio_path: Path to save the extracted audio.
    :return: Path to the extracted audio file.
    """
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    print(f"Extracted audio to {audio_path}")
    return audio_path

# ==================================
# Real-Time Filtering Logic
# ==================================

def process_video_stream(video_path, preferences):
    """
    Process video stream for real-time content filtering.
    :param video_path: Path to the video or stream URL.
    :param preferences: User preferences for filtering.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    net, classes = load_yolo_model()

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply filters based on preferences
        frame = apply_filters(frame, preferences, net, classes, cap, current_frame)
        current_frame += 1

        # Display the processed frame
        cv2.imshow('Filtered Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==================================
# Web API Endpoints
# ==================================

@app.route('/set_preferences', methods=['POST'])
def set_preferences():
    """
    Endpoint to set user preferences.
    :return: Confirmation of preferences saved.
    """
    data = request.json
    preferences = data.get("preferences", {})

    # Save preferences to a local file for persistence
    with open('preferences.json', 'w') as f:
        json.dump(preferences, f)

    return jsonify({"message": "Preferences saved successfully!", "preferences": preferences})

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    Endpoint to process video for filtering.
    :return: Stream processed video (placeholder).
    """
    video_path = request.json.get('video_path')

    # Load preferences from file
    if os.path.exists('preferences.json'):
        with open('preferences.json', 'r') as f:
            preferences = json.load(f)
    else:
        return jsonify({"error": "Preferences not set. Please set preferences first."}), 400

    if not os.path.exists(video_path):
        return jsonify({"error": "Video file does not exist."}), 400

    process_video_stream(video_path, preferences)
    return jsonify({"message": "Video processing started."})

# ==================================
# Main Entry
# ==================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        preferences = {}

        if os.path.exists('preferences.json'):
            with open('preferences.json', 'r') as f:
                preferences = json.load(f)

        process_video_stream(video_path, preferences)
    else:
        app.run(debug=True)
