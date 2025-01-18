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

# Import MouseInfoWindow
try:
    from mouseinfo import MouseInfoWindow
except ImportError:
    print("The 'mouseinfo' library or file is required. Please ensure it is in the same directory or installed.")

# Pre-Trained Models for Video Classification
model = keras.applications.MobileNetV2(weights="imagenet")

# NLP Model for Objectionable Language Detection
nlp = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", top_k=None)

# Define Default Objectionable Words or Phrases
DEFAULT_OBJECTIONABLE_WORDS = ["swearword1", "swearword2", "violent phrase", "suggestive phrase"]
CATEGORY_FILTERS = {
    "Profanity": ["swearword1", "swearword2", "f-word", "s-word"],
    "Violence": ["fight", "kill", "gun"],
    "Sexual Content": ["nude", "sex", "explicit"]
}

PREFERENCES_FILE = "user_preferences.json"

# Force TensorFlow to use CPU only
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
        MouseInfoWindow()
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
    def __init__(self, root):
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
        tk.Button(root, text="Mouse Info Tool", command=self.launch_mouseinfo).grid(row=11, column=0, pady=10)

        self.load_preferences()

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[["Video Files", "*.mp4;*.avi;*.mkv"]])
        if path:
            self.video_path.set(path)

    def browse_audio(self):
        path = filedialog.askopenfilename(filetypes=[["Audio Files", "*.wav"]])
        if path:
            self.audio_path.set(path)

    def browse_subtitles(self):
        path = filedialog.askopenfilename(filetypes=[["Subtitle Files", "*.srt"]])
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
        try:
            MouseInfoWindow()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Mouse Info Tool: {e}")

def main():
    if os.environ.get("DISPLAY"):
        root = tk.Tk()
        app = ContentFilterApp(root)
        root.mainloop()
    else:
        print("No graphical environment detected. Running in CLI mode.")
        cli_mode()

if __name__ == "__main__":
    main()
