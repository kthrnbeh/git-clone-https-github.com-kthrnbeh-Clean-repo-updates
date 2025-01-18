import cv2
from tensorflow import keras
import numpy as np
import speech_recognition as sr
from transformers import pipeline
import re
import time
import wave
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import argparse

# Optional imports with error handling
try:
    from pytube import YouTube, Playlist
except ImportError:
    print("The 'pytube' library is required. Please install it using `pip install pytube`.")

try:
    from streamlink import Streamlink
except ImportError:
    print("The 'streamlink' library is required. Please install it using `pip install streamlink`.")

try:
    import pyautogui
except ImportError:
    print("The 'pyautogui' library is required. Please install it using `pip install pyautogui`.")

# Import MouseInfoWindow with error handling
try:
    from mouseinfo import MouseInfoWindow
except ImportError:
    print("The 'mouseinfo' library is required. Please ensure it is installed.")

# Pre-Trained Models for Video Classification
model = keras.applications.MobileNetV2(weights="imagenet")

# NLP Model for Objectionable Language Detection
nlp = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", top_k=None)

# Default Objectionable Words or Phrases
DEFAULT_OBJECTIONABLE_WORDS = ["swearword1", "swearword2", "violent phrase", "suggestive phrase"]
CATEGORY_FILTERS = {
    "Profanity": ["swearword1", "swearword2", "f-word", "s-word"],
    "Violence": ["fight", "kill", "gun"],
    "Sexual Content": ["nude", "sex", "explicit"],
}

PREFERENCES_FILE = "user_preferences.json"

# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# CLI Mode Implementation
def cli_mode():
    parser = argparse.ArgumentParser(description="Content Filtering Application (CLI Mode)")
    parser.add_argument("--video", help="Path to the video file")
    parser.add_argument("--audio", help="Path to the audio file", required=False)
    parser.add_argument("--subtitles", help="Path to the subtitles file", required=False)
    parser.add_argument("--youtube", help="YouTube URL to download and process", required=False)
    parser.add_argument("--categories", help="Comma-separated categories to filter", required=False)
    parser.add_argument("--mode", help="Filtering mode: skip/mute/log", default="log")
    parser.add_argument("--mouseinfo", action="store_true", help="Launch Mouse Info Too

