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

