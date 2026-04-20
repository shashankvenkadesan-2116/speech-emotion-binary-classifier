# speech-emotion-binary-classifier
Speech Emotion Detector
An end-to-end Machine Learning application that classifies human emotions from speech using Digital Signal Processing (DSP) and Support Vector Machines (SVM). This project was developed as part of my curriculum focusing on Signal Processing and ML (Units IV–V).

Overview
This application allows users to upload .wav audio files (specifically designed for the RAVDESS format) and receive an instant classification of the emotional state. It distinguishes between Negative Emotions (Angry, Fearful, Disgusted, Surprised) and Neutral/Positive Emotions (Calm, Happy, Neutral).

Technical Stack
Frontend: HTML5, CSS3 (Modern UI with Glow effects/Grid background), JavaScript (Fetch API).

Backend: Flask (Python).

Signal Processing: Librosa, SciPy (Butterworth Filter).

Machine Learning: Scikit-learn (SVM Classifier), NumPy.

Key Technical Features
Signal Pre-processing: Implements a Butterworth bandpass filter (300Hz to 3500Hz) to isolate human speech frequencies and reduce noise.

Feature Extraction: Utilizes Mel-Frequency Cepstral Coefficients (MFCCs) with a Hamming window to extract 40 acoustic features from the filtered audio.

SVM Classification: Uses a Support Vector Classifier with a Polynomial kernel (degree 3) trained on the RAVDESS dataset.

Real-time Prediction: A Flask API handles file uploads, processes the signal in-memory, and returns the prediction with a confidence percentage.

Project Structure
Plaintext
├── app.py              # Flask Backend & ML Pipeline
├── index.html          # Modern Frontend UI
├── emotion_model.pkl   # Pre-trained SVM Model (generated after first run)
└── README.md           # Project Documentation
Installation & Setup
Clone the Repository

Bash
git clone https://github.com/yourusername/speech-emotion-detector.git
cd speech-emotion-detector
Install Dependencies

Bash
pip install flask flask-cors numpy librosa scipy scikit-learn
Configure Data Path
In app.py, update the DATA_DIRECTORY variable to point to your local RAVDESS dataset if you wish to retrain the model.

Run the Application

Start the Flask server:

Bash
python app.py
Open index.html in your preferred web browser.

Dataset Reference
This project is optimized for the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. It specifically filters for negative codes ['04', '05', '06', '08'] to distinguish emotional valence.
