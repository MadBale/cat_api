from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa
import os

app = Flask(__name__)
model = tf.lite.Interpreter(model_path="cat_sound_classifier.tflite")
model.allocate_tensors()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']

        # Save temporarily for debug
        file.save("temp.wav")

        audio_data, sr = librosa.load("temp.wav", sr=22050)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfcc = librosa.util.fix_length(mfcc, size=128, axis=1)
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc = np.expand_dims(mfcc, axis=0).astype(np.float32)

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.set_tensor(input_details[0]['index'], mfcc)
        model.invoke()
        output_data = model.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)

        labels = ["Neutral", "Annoyed", "Excited", "Hungry", "Calm", "Cautious", "Happy", "Angry", "Playful", "Sad", "In Pain", "Affectionate"]
        predicted_label = labels[prediction]

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        print("🔥 Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
