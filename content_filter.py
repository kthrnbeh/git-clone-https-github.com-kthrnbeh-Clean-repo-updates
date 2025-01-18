import cv2
import tf_keras as keras
import numpy as np
import speech_recognition as sr
from transformers import pipeline
import re
import time
import wave
import os

# Pre-Trained Models for Video Classification
model = keras.applications.MobileNetV2(weights="imagenet")

# NLP Model for Objectionable Language Detection
nlp = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", return_all_scores=True)

# Define Default Objectionable Words or Phrases
DEFAULT_OBJECTIONABLE_WORDS = ["swearword1", "swearword2", "violent phrase", "suggestive phrase"]
CATEGORY_FILTERS = {
    "Profanity": ["swearword1", "swearword2", "f-word", "s-word"],
    "Violence": ["fight", "kill", "gun"],
    "Sexual Content": ["nude", "sex", "explicit"]
}

def get_user_preferences():
    """Allow users to define their preferences for filtering."""
    print("Welcome to the Content Filtering System Configuration.")
    print("Choose your preferred filtering options:")

    print("1. Filtering Modes:")
    print("   a. Skip (Fast-forward objectionable content)")
    print("   b. Mute (Mute objectionable audio)")
    print("   c. Log Only (Log content but take no action)")
    mode = input("Select mode (a/b/c): ").strip().lower()

    print("\n2. Select Categories to Filter:")
    for i, category in enumerate(CATEGORY_FILTERS.keys(), 1):
        print(f"   {i}. {category}")
    selected_categories = input("Enter category numbers separated by commas (e.g., 1,2): ").strip()

    selected_words = []
    try:
        selected_indices = [int(x.strip()) - 1 for x in selected_categories.split(",")]
        for idx in selected_indices:
            category = list(CATEGORY_FILTERS.keys())[idx]
            selected_words.extend(CATEGORY_FILTERS[category])
    except (ValueError, IndexError):
        print("Invalid selection. Using default filters.")
        selected_words = DEFAULT_OBJECTIONABLE_WORDS

    print("\n3. Add Custom Objectionable Words (comma-separated, optional):")
    custom_words = input("Enter words (leave blank for default): ").strip()

    # Validate and store user preferences
    filtering_mode = mode if mode in ['a', 'b', 'c'] else 'a'
    objectionable_words = selected_words + ([word.strip() for word in custom_words.split(',')] if custom_words else [])

    return {
        "mode": filtering_mode,
        "words": objectionable_words
    }

# Initialize user preferences
USER_PREFERENCES = get_user_preferences()
OBJECTIONABLE_WORDS = USER_PREFERENCES["words"]

# Frame Processing

def preprocess_frame(frame):
    """Preprocess the frame for AI model prediction."""
    frame = cv2.resize(frame, (224, 224))
    frame = keras.applications.mobilenet_v2.preprocess_input(frame)
    return np.expand_dims(frame, axis=0)

def classify_frame(frame):
    """Classify a single frame and return detected labels."""
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    decoded_predictions = keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    return decoded_predictions

# Audio and Subtitle Processing

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file}")
        return ""
    except wave.Error:
        print(f"Invalid audio file format: {audio_file}")
        return ""
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def detect_objectionable_content(transcript):
    detected = []
    for word in OBJECTIONABLE_WORDS:
        if re.search(rf"\b{word}\b", transcript, re.IGNORECASE):
            detected.append(word)

    if not detected:
        nlp_results = nlp(transcript)
        for result in nlp_results[0]:
            if result['label'] in ["anger", "disgust"] and result['score'] > 0.8:
                detected.append(result['label'])
    return detected

# Sequential Processing

def process_video_sequentially(video_path, audio_path, subtitles_path):
    """Process video, audio, and subtitles sequentially for simplicity."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("sequential_output.mp4", fourcc, fps, (width, height))

    # Process audio and subtitles
    transcript = transcribe_audio(audio_path)
    audio_detections = detect_objectionable_content(transcript)

    if os.path.exists(subtitles_path):
        with open(subtitles_path, "r") as subtitle_file:
            subtitles = subtitle_file.read()
            subtitle_detections = detect_objectionable_content(subtitles)
    else:
        subtitle_detections = []

    print("Audio Detections:", audio_detections)
    print("Subtitle Detections:", subtitle_detections)

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Classify frame
        predictions = classify_frame(frame)
        labels = [label[1] for label in predictions[0]]

        action_needed = False
        if "nude" in labels or "violence" in labels or audio_detections or subtitle_detections:
            action_needed = True

        if action_needed:
            if USER_PREFERENCES["mode"] == 'a':
                print("Skipping objectionable scene.")
                continue
            elif USER_PREFERENCES["mode"] == 'b':
                print("Muting objectionable content.")

        out.write(frame)
        cv2.imshow("Processed Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print(f"Processing complete in {end_time - start_time:.2f} seconds.")

# Create a Sample Audio File
if not os.path.exists("input_audio.wav"):
    import wave
    import struct

    # Generate a dummy .wav file
    sample_rate = 44100.0  # Hertz
    duration = 2.0  # seconds
    frequency = 440.0  # Hertz

    n_samples = int(sample_rate * duration)
    wav_file = wave.open("input_audio.wav", "w")
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16 bits
    wav_file.setframerate(int(sample_rate))

    for i in range(n_samples):
        value = int(32767.0 * np.sin(2.0 * np.pi * frequency * i / sample_rate))
        data = struct.pack('<h', value)
        wav_file.writeframesraw(data)

    wav_file.close()

# Example Usage
video_path = "input_video.mp4"
audio_path = "input_audio.wav"
subtitles_path = "input_subtitles.srt"
process_video_sequentially(video_path, audio_path, subtitles_path)
