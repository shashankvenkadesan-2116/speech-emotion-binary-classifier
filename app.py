"""
Flask backend for Speech Emotion Detector
Run this file: python app.py
Then open index.html in your browser.
"""

import os
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML page

# ─── MODEL PATH ───────────────────────────────────────────────────────────────
# Point this to your dataset to train the model (only needed once)
DATA_DIRECTORY = r"C:\Users\DELL\OneDrive\Documents\PROJECT\data"
MODEL_FILE = "emotion_model.pkl"

# ─── SIGNAL PROCESSING ────────────────────────────────────────────────────────
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    filtered_audio = butter_bandpass_filter(audio, 300, 3500, sample_rate, order=5)
    mfccs = librosa.feature.mfcc(
        y=filtered_audio,
        sr=sample_rate,
        n_mfcc=40,
        win_length=int(0.025 * sample_rate),
        hop_length=int(0.010 * sample_rate),
        window='hamming'
    )
    return np.mean(mfccs.T, axis=0)

# ─── TRAIN & SAVE MODEL ───────────────────────────────────────────────────────
def train_and_save_model():
    neg_codes = ['04', '05', '06', '08']
    features, labels = [], []

    print("Training model from dataset...")
    for root, dirs, files in os.walk(DATA_DIRECTORY):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion = parts[2]
                    label = 1 if emotion in neg_codes else 0
                    try:
                        f = extract_features(os.path.join(root, file))
                        features.append(f)
                        labels.append(label)
                    except Exception as e:
                        print(f"Skipping {file}: {e}")

    X, y = np.array(features), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel='poly', degree=3, C=1.0, probability=True)
    model.fit(X_train, y_train)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to {MODEL_FILE}")
    return model

# ─── LOAD OR TRAIN MODEL ──────────────────────────────────────────────────────
if os.path.exists(MODEL_FILE):
    print(f"Loading saved model from {MODEL_FILE}")
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
else:
    model = train_and_save_model()

# ─── API ENDPOINT ─────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.wav'):
        return jsonify({'error': 'Only .wav files are supported'}), 400

    # Save temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        label = "Negative Emotion" if prediction == 1 else "Neutral / Positive"
        confidence = float(max(probability)) * 100

        return jsonify({
            'prediction': int(prediction),
            'label': label,
            'confidence': round(confidence, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        train_and_save_model()
        return jsonify({'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)
