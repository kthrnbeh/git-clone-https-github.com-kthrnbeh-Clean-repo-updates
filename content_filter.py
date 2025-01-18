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

        # Filtering Mode
        tk.Label(root, text="Select Filtering Mode:").grid(row=3, column=0, sticky="w")
        self.filter_mode = tk.StringVar(value="a")
        tk.Radiobutton(root, text="Skip", variable=self.filter_mode, value="a").grid(row=3, column=1, sticky="w")
        tk.Radiobutton(root, text="Mute", variable=self.filter_mode, value="b").grid(row=3, column=1)
        tk.Radiobutton(root, text="Log Only", variable=self.filter_mode, value="c").grid(row=3, column=1, sticky="e")

        # Category Selection
        tk.Label(root, text="Select Categories to Filter:").grid(row=4, column=0, sticky="w")
        self.category_vars = {category: tk.BooleanVar(value=False) for category in CATEGORY_FILTERS.keys()}
        for i, category in enumerate(CATEGORY_FILTERS.keys()):
            tk.Checkbutton(root, text=category, variable=self.category_vars[category]).grid(row=4 + i // 3, column=i % 3 + 1, sticky="w")

        # Progress Bar
        tk.Label(root, text="Processing Progress:").grid(row=7, column=0, sticky="w")
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.grid(row=7, column=1, columnspan=2, pady=10)

        # Save Preferences
        tk.Button(root, text="Save Preferences", command=self.save_preferences).grid(row=8, column=0, pady=10)

        # Load Preferences
        tk.Button(root, text="Load Preferences", command=self.load_preferences).grid(row=8, column=1, pady=10)

        # Process Button
        tk.Button(root, text="Start Filtering", command=self.start_filtering).grid(row=8, column=2, pady=10)

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

    def start_filtering(self):
        video_path = self.video_path.get()
        audio_path = self.audio_path.get()
        subtitles_path = self.subtitles_path.get()

        if not video_path or not audio_path or not subtitles_path:
            messagebox.showerror("Error", "Please select all required files.")
            return

        # Get user-selected categories and filtering mode
        selected_categories = [cat for cat, var in self.category_vars.items() if var.get()]
        objectionable_words = []
        for category in selected_categories:
            objectionable_words.extend(CATEGORY_FILTERS[category])

        filtering_mode = self.filter_mode.get()

        # Save preferences and process
        self.process_video_sequentially(video_path, audio_path, subtitles_path, filtering_mode, objectionable_words)

    def process_video_sequentially(self, video_path, audio_path, subtitles_path, filtering_mode, objectionable_words):
        """Process video, audio, and subtitles sequentially."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("sequential_output.mp4", fourcc, fps, (width, height))

        # Process audio and subtitles
        transcript = self.transcribe_audio(audio_path)
        audio_detections = self.detect_objectionable_content(transcript, objectionable_words)

        if os.path.exists(subtitles_path):
            with open(subtitles_path, "r") as subtitle_file:
                subtitles = subtitle_file.read()
                subtitle_detections = self.detect_objectionable_content(subtitles, objectionable_words)
        else:
            subtitle_detections = []

        print("Audio Detections:", audio_detections)
        print("Subtitle Detections:", subtitle_detections)

        start_time = time.time()
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Classify frame
            predictions = self.classify_frame(frame)
            labels = [label[1] for label in predictions[0]]

            action_needed = False
            if "nude" in labels or "violence" in labels or audio_detections or subtitle_detections:
                action_needed = True

            if action_needed:
                if filtering_mode == 'a':
                    print("Skipping objectionable scene.")
                    continue
                elif filtering_mode == 'b':
                    print("Muting objectionable content.")

            out.write(frame)

            # Update progress bar
            current_frame += 1
            progress = (current_frame / total_frames) * 100
            self.progress_bar['value'] = progress
            self.root.update_idletasks()

        cap.release()
        out.release()

        end_time = time.time()
        messagebox.showinfo("Success", f"Processing complete in {end_time - start_time:.2f} seconds.")

    def classify_frame(self, frame):
        frame = cv2.resize(frame, (224, 224))
        frame = keras.applications.mobilenet_v2.preprocess_input(frame)
        processed_frame = np.expand_dims(frame, axis=0)
        predictions = model.predict(processed_frame, verbose=0)
        return keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

    def transcribe_audio(self, audio_file):
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

    def detect_objectionable_content(self, transcript, objectionable_words):
        detected = []
        for word in objectionable_words:
            if re.search(rf"\b{word}\b", transcript, re.IGNORECASE):
                detected.append(word)

        if not detected:
            nlp_results = nlp(transcript)
            for result in nlp_results[0]:
                if result['label'] in ["anger", "disgust"] and result['score'] > 0.8:
                    detected.append(result['label'])
        return detected

if __name__ == "__main__":
    root = tk.Tk()
    app = ContentFilterApp(root)
    root.mainloop()
