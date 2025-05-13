from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
import os

app = Flask(__name__)

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="cat_sound_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example labels (replace with your actual label list)
LABELS = ["Hungry", "Playful", "Annoyed", "Happy", "Sad", "Tired", "Scared", "Angry", "Curious", "Neutral", "Cautious", "Excited"]

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    mfcc = mfcc[:128, :128]  # Ensure shape is 128x128
    mfcc = np.expand_dims(mfcc, axis=(0, -1))  # Shape: [1, 128, 128, 1]
    return mfcc.astype(np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    temp_path = "temp.wav"
    audio_file.save(temp_path)

    try:
        input_tensor = preprocess_audio(temp_path)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        top_idx = int(np.argmax(output_data))
        result = {
            "label": LABELS[top_idx],
            "confidence": float(output_data[top_idx])
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(temp_path)

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render sets PORT env var
    app.run(host='0.0.0.0', port=port)
