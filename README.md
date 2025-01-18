Content Filter Application

Overview

The Content Filter Application is designed to provide automated filtering of objectionable content from video, audio, and subtitle files. It is highly configurable, allowing users to define preferences for content categories and filtering modes, and supports both GUI and CLI environments.

Features

Filter Modes: Skip, Mute, or Log objectionable content.

Category Filtering: Profanity, violence, sexual content, and customizable categories.

YouTube Integration: Download and process videos or playlists directly.

User Preferences: Save and load filtering configurations.

Environment Detection: Automatically switches between GUI and CLI modes.

Subtitle and Audio Processing: Scan and filter objectionable words and phrases in text and spoken content.

Requirements

Python 3.8+

Required Python libraries:

tensorflow

keras

transformers

pytube

streamlink

pyautogui

speech_recognition

opencv-python

numpy

ttkbootstrap

Install dependencies using:

pip install -r requirements.txt

Setup

Clone the repository:

git clone https://github.com/your-repo/content-filter.git
cd content-filter

Create a virtual environment:

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate   # Windows

Install dependencies:

pip install -r requirements.txt

Usage

GUI Mode

Run the application in a graphical environment:

python content_filter.py

CLI Mode

Run the application in a headless setup:

python content_filter.py --video path_to_video --audio path_to_audio --subtitles path_to_subtitles --categories Profanity,Violence --mode skip

YouTube Integration

To process YouTube videos or playlists, provide the URL:

python content_filter.py --youtube https://www.youtube.com/your_video_or_playlist_url

Options

--video: Path to the video file.

--audio: Path to the audio file.

--subtitles: Path to the subtitle file.

--categories: Comma-separated list of categories to filter (e.g., Profanity, Violence).

--mode: Filtering mode (skip, mute, or log).

--youtube: YouTube video or playlist URL.

Saving and Loading Preferences

In the GUI, use the "Save Preferences" and "Load Preferences" buttons to store or retrieve your settings.

In CLI mode, preferences are automatically loaded from user_preferences.json if available.

Testing and Troubleshooting

Run tests with various inputs in both GUI and CLI modes to ensure all functionalities work as expected.

Common Issues:

Missing Libraries: Ensure all required libraries are installed (pip install -r requirements.txt).

YouTube Download Errors: Verify the YouTube URL and check network connectivity.

Graphical Environment: If the GUI does not launch, ensure a display environment is available.

Contributing

Feel free to fork the repository and submit pull requests for new features or bug fixes. Ensure your code adheres to the repository's style guide.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements

Special thanks to:

Hugging Face for the transformers library.

TensorFlow and Keras for model integration.

Contributors and open-source projects that made this possible.

mport cv2
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

Conditionally import MouseInfoWindow

try:
if os.environ.get("DISPLAY") or os.name == "nt":  # Check for GUI availability
from mouseinfo import MouseInfoWindow
GUI_AVAILABLE = True
else:
GUI_AVAILABLE = False
except ImportError:
GUI_AVAILABLE = False
print("MouseInfo is not available in this environment.")

Pre-Trained Models for Video Classification

model = keras.applications.MobileNetV2(weights="imagenet")

NLP Model for Objectionable Language Detection

nlp = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", top_k=None)

Define Default Objectionable Words or Phrases

DEFAULT_OBJECTIONABLE_WORDS = ["swearword1", "swearword2", "violent phrase", "suggestive phrase"]
CATEGORY_FILTERS = {
"Profanity": ["swearword1", "swearword2", "f-word", "s-word"],
"Violence": ["fight", "kill", "gun"],
"Sexual Content": ["nude", "sex", "explicit"]
}

PREFERENCES_FILE = "user_preferences.json"

Force TensorFlow to use CPU only

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def cli_mode():
parser = argparse.ArgumentParser(description="Content Filtering Application (CLI Mode)")
parser.add_argument("--video", help="Path to the video file")
parser.add_argument("--audio", help="Path to the audio file", required=False)
parser.add_argument("--subtitles", help="Path to the subtitles file", required=False)
parser.add_argument("--youtube", help="YouTube URL to download and process", required=False)
parser.add_argument("--categories", help="Comma-separated categories to filter", required=False)
parser.add_argument("--mode", help="Filtering mode: skip/mute/log", default="log")
parser.add_argument("--mouseinfo", action="store_true", help="Launch Mouse Info Tool")

args = parser.parse_args()

if args.mouseinfo:
    if GUI_AVAILABLE:
        MouseInfoWindow()
    else:
        print("Mouse Info Tool is not available in this environment.")
    return

if args.youtube:
    print("Downloading YouTube content...")
    # Add YouTube download logic here
elif args.video:
    print(f"Processing video: {args.video}")
    # Add local file processing logic here
else:
    print("No valid input provided. Use --help for usage details.")

class ContentFilterApp:
def init(self, root):
self.root = root
self.root.title("Content Filtering System")

    # File Selection
    self.video_path = tk.StringVar()
    self.audio_path = tk.StringVar()
    self.subtitles_path = tk.StringVar()

    tk.Label(root, text="Select Video File:").grid(row=0, column=0, sticky="w")
    tk.Entry(root, textvariable=self.video_path, width=50).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=self.browse_video).grid(row=0, column=2)

    tk.Label(root, text="Select Audio File:").grid(row=1, column=0, sticky="w")
    tk.Entry(root, textvariable=self.audio_path, width=50).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=self.browse_audio).grid(row=1, column=2)

    tk.Label(root, text="Select Subtitles File:").grid(row=2, column=0, sticky="w")
    tk.Entry(root, textvariable=self.subtitles_path, width=50).grid(row=2, column=1)
    tk.Button(root, text="Browse", command=self.browse_subtitles).grid(row=2, column=2)

    # YouTube URL Download Section
    tk.Label(root, text="Enter YouTube Video/Playlist URL:").grid(row=3, column=0, sticky="w")
    self.youtube_url = tk.StringVar()
    tk.Entry(root, textvariable=self.youtube_url, width=50).grid(row=3, column=1)
    tk.Button(root, text="Download", command=self.download_youtube_content).grid(row=3, column=2)

    # Source Type Selection
    tk.Label(root, text="Select Source Type:").grid(row=4, column=0, sticky="w")
    self.source_type = tk.StringVar(value="Local File")
    tk.OptionMenu(root, self.source_type, "Local File", "YouTube", "DVD", "Live Stream").grid(row=4, column=1)

    # Filtering Mode
    tk.Label(root, text="Select Filtering Mode:").grid(row=5, column=0, sticky="w")
    self.filter_mode = tk.StringVar(value="a")
    tk.Radiobutton(root, text="Skip", variable=self.filter_mode, value="a").grid(row=5, column=1, sticky="w")
    tk.Radiobutton(root, text="Mute", variable=self.filter_mode, value="b").grid(row=5, column=1)
    tk.Radiobutton(root, text="Log Only", variable=self.filter_mode, value="c").grid(row=5, column=1, sticky="e")

    # Category Selection
    tk.Label(root, text="Select Categories to Filter:").grid(row=6, column=0, sticky="w")
    self.category_vars = {category: tk.BooleanVar(value=False) for category in CATEGORY_FILTERS.keys()}
    for i, category in enumerate(CATEGORY_FILTERS.keys()):
        tk.Checkbutton(root, text=category, variable=self.category_vars[category]).grid(row=6 + i // 3, column=i % 3 + 1, sticky="w")

    # Progress Bar
    tk.Label(root, text="Processing Progress:").grid(row=9, column=0, sticky="w")
    self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
    self.progress_bar.grid(row=9, column=1, columnspan=2, pady=10)

    # Save Preferences
    tk.Button(root, text="Save Preferences", command=self.save_preferences).grid(row=10, column=0, pady=10)

    # Load Preferences
    tk.Button(root, text="Load Preferences", command=self.load_preferences).grid(row=10, column=1, pady=10)

    # Process Button
    tk.Button(root, text="Start Filtering", command=self.start_filtering).grid(row=10, column=2, pady=10)

    # Mouse Info Tool Button
    if GUI_AVAILABLE:
        tk.Button(root, text="Mouse Info Tool", command=self.launch_mouseinfo).grid(row=11, column=0, pady=10)

    self.load_preferences()

def browse_video(self):
    path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
    if path:
        self.video_path.set(path)

def browse_audio(self):
    path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if path:
        self.audio_path.set(path)

def browse_subtitles(self):
    path = filedialog.askopenfilename(filetypes=[("Subtitle Files", "*.srt")])
    if path:
        self.subtitles_path.set(path)

def download_youtube_content(self):
    url = self.youtube_url.get()
    if not url:
        messagebox.showerror("Error", "Please enter a valid YouTube URL.")
        return

    try:
        if "playlist" in url:
            playlist = Playlist(url)
            download_path = "downloads/playlist"
            os.makedirs(download_path, exist_ok=True)
            for video in playlist.videos:
                video.streams.filter(progressive=True, file_extension="mp4").first().download(download_path)
            messagebox.showinfo("Success", f"Playlist downloaded to: {download_path}")
        else:
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension="mp4").first()
            download_path = "downloads"
            os.makedirs(download_path, exist_ok=True)
            file_path = stream.download(download_path)
            messagebox.showinfo("Success", f"Video downloaded to: {file_path}")
            self.video_path.set(file_path)  # Set the downloaded video for processing
    except Exception as e:
        messagebox.showerror("Error", f"Failed to download video/playlist: {e}")

def save_preferences(self):
    preferences = {
        "filter_mode": self.filter_mode.get(),
        "categories": {cat: var.get() for cat, var in self.category_vars.items()}
    }
    with open(PREFERENCES_FILE, "w") as f:
        json.dump(preferences, f)
    messagebox.showinfo("Success", "Preferences saved successfully.")

def load_preferences(self):
    if os.path.exists(PREFERENCES_FILE):
        with open(PREFERENCES_FILE, "r") as f:
            preferences = json.load(f)
        self.filter_mode.set(preferences.get("filter_mode", "a"))
        for cat, value in preferences.get("categories", {}).items():
            if cat in self.category_vars:
                self.category_vars[cat].set(value)

def launch_mouseinfo(self):
    if GUI_AVAILABLE:
        try:
            MouseInfoWindow()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Mouse Info Tool: {e}")
    else:
        messagebox.showerror("Error", "Mouse Info Tool is not available in this environment.")

def main():
if os.environ.get("DISPLAY") or os.name == "nt":  # Check for GUI availability
try:
root = tk.Tk()
app = ContentFilterApp(root)
root.mainloop()
except Exception as e:
print(f"Failed to launch GUI mode: {e}")
print("Falling back to CLI mode.")
cli_mode()
else:
print("No graphical environment detected. Running in CLI mode.")
cli_mode()

if name == "main":
main()

