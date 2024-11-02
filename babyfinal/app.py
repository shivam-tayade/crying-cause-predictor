from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import librosa
import os

app = Flask(__name__)

# Load the model and the label encoder
model = tf.keras.models.load_model('model/model.h5')
label_encoder = np.load('model/classes.npy')

# Feature extraction function (from the notebook)
def extract_features_from_audio(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)

    features = np.concatenate((
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0),
        np.mean(contrast.T, axis=0)
    ))

    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file to a temporary location
    temp_path = os.path.join('uploads', file.filename)
    file.save(temp_path)

    try:
        # Load and process the audio file
        print("Loading audio file")
        audio, sample_rate = librosa.load(temp_path, res_type='kaiser_fast')
        
        # Extract features
        print("Extracting features")
        features = extract_features_from_audio(audio, sample_rate)
        features = features.reshape(1, -1)  # Reshape for model input

        # Make a prediction
        print("Making prediction")
        prediction = model.predict(features)
        print("Prediction raw output:", prediction)
        
        predicted_label = label_encoder[np.argmax(prediction)]
        print("Predicted label:", predicted_label)

        # Return the result
        return jsonify({'result': predicted_label})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)})

    finally:
        # Clean up the temporary file
        os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)
