 #Speech Emotion Detector
##Project Overview
This end-to-end Machine Learning application classifies human emotions from speech using Digital Signal Processing (DSP) and Support Vector Machines (SVM). Developed as part of a curriculum focusing on Signal Processing and ML (Units IV–V), the tool allows users to upload .wav audio files and receive instant emotional classification.

It distinguishes between:

Negative Emotions: Angry, Fearful, Disgusted, and Surprised.

Neutral/Positive Emotions: Calm, Happy, and Neutral.

##Technical Stack
Frontend: HTML5, CSS3 (Modern UI with Glow effects/Grid background), and JavaScript using the Fetch API.

Backend: Python-based Flask server.

Signal Processing: Librosa and SciPy (specifically for Butterworth filtering).

Machine Learning: Scikit-learn (SVM Classifier) and NumPy.

##Key Technical Features
Signal Pre-processing: Implements a Butterworth bandpass filter (300Hz to 3500Hz) to isolate human speech and reduce ambient noise.

Feature Extraction: Utilizes Mel-Frequency Cepstral Coefficients (MFCCs) with a Hamming window to extract 40 specific acoustic features.

SVM Classification: Features a Support Vector Classifier with a Polynomial kernel (degree 3) trained on the RAVDESS dataset.

Real-time Prediction: A custom Flask API handles in-memory signal processing to return predictions with a confidence percentage.

## Project Structure
Plaintext
├── app.py              # Flask Backend & ML Pipeline
├── index.html          # Modern Frontend UI
├── emotion_model.pkl   # Pre-trained SVM Model (auto-generated)
└── README.md           # Project Documentation
## Installation & Setup
1. Clone the Repository

Bash
git clone https://github.com/yourusername/speech-emotion-detector.git
cd speech-emotion-detector
2. Install Dependencies

Bash
pip install flask flask-cors numpy librosa scipy scikit-learn
3. Configure Data Path
In app.py, update the DATA_DIRECTORY variable to point to your local dataset if you intend to retrain the model.

4. Run the Application

Start the server: python app.py

Launch UI: Open index.html in your web browser.

## Dataset Reference
This project is optimized for the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The logic specifically filters for negative codes ['04', '05', '06', '08'] to distinguish emotional valence.
