import os
import logging  # For logging

try:
    import cv2  # For Computer Vision
except ImportError as e:
    logging.error("Error: OpenCV is not installed. Please install it using 'pip install opencv-python'.")
    raise e

try:
    from transformers import pipeline  # For NLP
except ImportError as e:
    logging.error("Error: Transformers module is not installed. Please install it using 'pip install transformers'.")
    raise e

try:
    import speech_recognition as sr  # For real-time audio transcription
except ImportError as e:
    logging.error("Error: SpeechRecognition module is not installed. Please install it using 'pip install SpeechRecognition'.")
    raise e

import numpy as np  # For array manipulation
from flask import Flask  # For Web API

try:
    from moviepy.editor import VideoFileClip  # For audio extraction
except ImportError as e:
    logging.error("Error: MoviePy module is not installed. Please install it using 'pip install moviepy'.")
    raise e

try:
    from pytube import YouTube  # For YouTube video downloading
except ImportError as e:
    logging.error("Error: PyTube module is not installed. Please install it using 'pip install pytube'.")
    raise e

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
    with open(class_labels_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, classes

# Detect objectionable content using YOLO
def detect_objectionable_content_yolo(frame, net, classes):
    """
    Detects objectionable content in a video frame using YOLO.
    :param frame: The video frame to analyze.
    :param net: The preloaded YOLO model.
    :param classes: Class labels used by YOLO.
    :return: True if objectionable content is detected, False otherwise.
    """
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["nude", "explicit"]:
                return True
    return False

# Transcribe and analyze audio for objectionable language
def transcribe_and_analyze_audio(audio):
    """
    Transcribes and analyzes audio for objectionable language.
    :param audio: Audio data to analyze.
    :return: True if objectionable content is detected, False otherwise.
    """
    recognizer = sr.Recognizer()
    try:
        transcription = recognizer.recognize_google(audio)
        nlp_model = pipeline("text-classification", model="distilbert-base-uncased")
        analysis = nlp_model(transcription)
        return any(item['label'] in ['HATE', 'PROFANITY'] for item in analysis)
    except sr.RequestError:
        logging.error("Error: Could not connect to speech recognition service")
    except sr.UnknownValueError:
        logging.error("Error: Could not understand audio")
    return False

# Apply video filters
def apply_filters(frame, preferences, net, classes, cap, current_frame):
    """
    Applies filters to a video frame based on user preferences.
    :param frame: The current video frame.
    :param preferences: User-defined preferences.
    :param net: Preloaded YOLO model.
    :param classes: Class labels.
    :param cap: Video capture object.
    :param current_frame: The current frame number.
    :return: Processed video frame.
    """
    if preferences.get('blur') and detect_objectionable_content_yolo(frame, net, classes):
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        logging.info("Blurred frame at timestamp %.2f seconds.", current_frame / cap.get(cv2.CAP_PROP_FPS))

    if preferences.get('skip') and detect_objectionable_content_yolo(frame, net, classes):
        skip_frames = 150
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + skip_frames)
        logging.info("Skipped frames at timestamp %.2f seconds.", current_frame / cap.get(cv2.CAP_PROP_FPS))

    return frame

# ==================================
# YouTube Video Download
# ==================================

def download_youtube_video(url, output_path="downloads"):
    """
    Downloads a YouTube video.
    :param url: YouTube video URL.
    :param output_path: Directory for saving the video.
    :return: Path to downloaded video.
    """
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return stream.download(output_path)

# ==================================
# Audio Extraction
# ==================================

def extract_audio(video_path, audio_path="output_audio.wav"):
    """
    Extracts audio from a video file.
    :param video_path: Path to video file.
    :param audio_path: Output audio file path.
    """
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    return audio_path
