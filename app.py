# Combine all three together
from dotenv import load_dotenv
from openai import OpenAI
import speech_recognition as sr
from gtts import gTTS
import sounddevice as sd
import soundfile as sf

from pydub import AudioSegment
from pydub.playback import play

import numpy as np
from io import BytesIO
import os
import time
## Need PyAudio as a dependency for the microphone to work
import pyaudio
#import whisper
# Need to import PyTorch for Whisper
import torch
import json

# FLASK!!!
from flask import Flask, jsonify, send_file, session, Response, jsonify, request, after_this_request, send_file, redirect, url_for
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename


from transformers import MarianMTModel, MarianTokenizer
import torch

# get current directory of script
# Define current working directory
current_dir = os.getcwd()


app = Flask(__name__)
# Enable CORS
cors = CORS(app)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("The file does not exist")


# upload_directory = "/uploads"

####
# This bottom code is for mac version
####
# Get the directory of the current file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Create the path to the uploads directory
upload_directory = os.path.join(current_directory, 'uploads')


if not os.path.exists(upload_directory):
    os.mkdir(upload_directory)


# Initial Delete
for f in os.listdir(upload_directory):
    os.remove(os.path.join(upload_directory, f))

# Set secret key in Flask
app.config['SECRET_KEY'] = 'x8D13XJ5+FTGNh5agRU7nF0B'    

# Max content length (64 MB)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = upload_directory


# Now, you can access the environment variables using the `os` module
import os
from speech_recognition import UnknownValueError
from speech_recognition import RequestError



# Model and tokenizer setup for both base and trained models
# Base Models
ENESbase = "Helsinki-NLP/opus-mt-en-es"
tokenizer_base_enes = MarianTokenizer.from_pretrained(ENESbase)
model_base_enes = MarianMTModel.from_pretrained(ENESbase)
device_base_enes = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_base_enes.to(device_base_enes)

ESENbase = "Helsinki-NLP/opus-mt-es-en"
tokenizer_base_esen = MarianTokenizer.from_pretrained(ESENbase)
model_base_esen = MarianMTModel.from_pretrained(ESENbase)
device_base_esen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_base_esen.to(device_base_esen)

# Trained Models
ENESTrained = "./ModelVersions/EN_ES_Iteration"
tokenizer_trained_enes = MarianTokenizer.from_pretrained(ENESTrained)
model_trained_enes = MarianMTModel.from_pretrained(ENESTrained)
device_trained_enes = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_trained_enes.to(device_trained_enes)

ESENTrained = "./ModelVersions/ES_EN_Iteration"
tokenizer_trained_esen = MarianTokenizer.from_pretrained(ESENTrained)
model_trained_esen = MarianMTModel.from_pretrained(ESENTrained)
device_trained_esen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_trained_esen.to(device_trained_esen)

def translate(text, source_lang, target_lang):
    prefix = f">>{target_lang}<< "
    prefixed_text = prefix + text

    try:
        if source_lang == "en" and target_lang == "es":
            # Base model EN-ES
            inputs_base = tokenizer_base_enes(prefixed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs_base = {k: v.to(device_base_enes) for k, v in inputs_base.items()}
            outputs_base = model_base_enes.generate(**inputs_base)

            # Trained model EN-ES
            inputs_trained = tokenizer_trained_enes(prefixed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs_trained = {k: v.to(device_trained_enes) for k, v in inputs_trained.items()}
            outputs_trained = model_trained_enes.generate(**inputs_trained)

        elif source_lang == "es" and target_lang == "en":
            # Base model ES-EN
            inputs_base = tokenizer_base_esen(prefixed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs_base = {k: v.to(device_base_esen) for k, v in inputs_base.items()}
            outputs_base = model_base_esen.generate(**inputs_base)

            # Trained model ES-EN
            inputs_trained = tokenizer_trained_esen(prefixed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs_trained = {k: v.to(device_trained_esen) for k, v in inputs_trained.items()}
            outputs_trained = model_trained_esen.generate(**inputs_trained)

        else:
            raise ValueError("Unsupported language pair")

        # Decode the tokens to strings
        translation_base = tokenizer_base_enes.decode(outputs_base[0], skip_special_tokens=True)
        translation_trained = tokenizer_trained_enes.decode(outputs_trained[0], skip_special_tokens=True)

        return translation_base, translation_trained

    except Exception as e:
        print(f"Error during translation: {e}")
        return None, None



# PYTHON AUDIO SETUP
# Test

# Initialize the recognizer
recognizer = sr.Recognizer()

# Testing function
def play_audio(file_path):
    try:
        # Read the audio file
        data, samplerate = sf.read(file_path)

        # Play the audio file
        sd.play(data, samplerate)
        sd.wait()

    except Exception as e:
        print(f"Error playing audio: {e}")


# Helper Method to delete files
# def delete_file(file_path):
#     """
#     Deletes a file at the specified path.

#     Args:
#         file_path (str): The path of the file to be deleted.
        
#     Returns:
#         bool: True if the file was successfully deleted, False otherwise.
#     """
#     if os.path.exists(file_path):
#         os.remove(file_path)
#         #print(f"The file at {file_path} has been successfully deleted.")
#         return True
#     else:
#         #print(f"The file at {file_path} does not exist.")
#         return False
    

translations = {}


# REST API ENDPOINT
@app.route('/speech-to-text', methods=['POST'])
@cross_origin()
def speech_to_text():
    # MOVED in order to define before use
    current_time = int(time.time() * 1000)
    #### RECEIVE AND PROCESS REQUEST
   
    #print(request.content_type)
    #print(request.content_length)
    #print(request.data)

    # Wrapper
    # Returns a FileStorage class which is a wrapper over incoming files
    # Assuming you're receiving the file in a form field called 'file'
    the_file = request.files['file']

    # Use a relative path for the uploads directory
    save_path = os.path.join('uploads', the_file.filename)

    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    the_file.save(save_path)

    # IMPORTANT
    # If the destination is a file object you have to close it yourself after the call. 
    the_file.close()

    selected_language = request.form.get('language')
    if selected_language == "Spanish":
        translate_to = "en"
    else:
        translate_to = "es"

    #### SPEECH TO TEXT

    # Load the WebM audio file
    audio = AudioSegment.from_file(save_path, format="webm")

    # Export the audio to WAV format for audio purposes
    # out_path = os.path.join(current_dir + "\\uploads", f"temp_{current_time}.wav")
    out_path = os.path.join(current_dir, "uploads", f"temp_{current_time}.wav")
    audio.export(out_path, format="wav")

    with sr.AudioFile(out_path) as source:
        audio = recognizer.record(source)  # read the entire audio file
        
    print("Recognizing...")
    # Recognize audio using Google Speech Recognition
    try:
        # Recognize audio using Google Speech Recognition
        if selected_language == "Spanish":
            text = recognizer.recognize_google(audio, language="es-MX")
        else:
            text = recognizer.recognize_google(audio, language="en-US")
        print("Recognized text:", text)

    except UnknownValueError:
        print("Google Speech Recognition could not understand audio")

    except RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    except TimeoutError:
        print("Google Speech Recognition request timed out")

    recognized_text = text

    source_lang = "es" if selected_language == "Spanish" else "en"
    target_lang = "en" if selected_language == "Spanish" else "es"
    base_text, trained_text  = translate(recognized_text, source_lang, target_lang)
    print(f"Base text: {base_text}\n Trained text: {trained_text}")


    # Determine the correct language code for gTTS based on the target language
    tts_lang_code = target_lang
    # tts = gTTS(text=translated_text, lang=translate_to, tld='com.mx')
    tts = gTTS(text=trained_text, lang=target_lang, tld='com.mx' if tts_lang_code == 'es' else 'com')

    # Save the speech as an in-memory BytesIO object
    audio_data = BytesIO()

    # Write bytes to a file-like object
    tts.write_to_fp(audio_data)

    # Write to an actual file
    # new_filename = os.path.join(current_dir + "\\uploads", f"new_{current_time}.wav")
    new_filename = os.path.join(current_dir, "uploads", f"new_{current_time}.wav")

    tts.save(new_filename)

    # Rewind the audio data
    audio_data.seek(0)

    #### RETURN THE AUDIO FILE & JSON

    # Delete temp files once done
    delete_file(save_path)
    delete_file(out_path)

    # return jsonify({'englishText': prompt, 'spanishText': text}), 200
    global translations # to access the translations dictionary
    translations = {
        'promptText': recognized_text,
        'BaseText': base_text,
        'TrainedText': trained_text
    }

    # Exits the Flask REST METHOD
    # Return actual AUDIO FILE ITSELF WITHOUT JSON
    # In HTTP Response -> Content-Disposition
    return send_file(
       new_filename, 
       mimetype="audio/wav", 
       as_attachment=False,
    )

# Additional endpoint to get translations as JSON
@app.route('/get_translation', methods=['GET'])
def get_translation():
    promptText = translations.get('promptText', 'No prompt text found')
    BaseText = translations.get('BaseText', 'No base model translation found')
    TrainedText = translations.get('TrainedText', 'No trained model translation found')
    
    return jsonify({
        'PromptText': promptText,
        'BaseText': BaseText,
        'TrainedText': TrainedText
    }), 200


# FOR SECURITY
# Remove the "Server" response header for all routes
@app.after_request
def remove_server_header(response):
    response.headers.pop('Server', None)
    return response



# For it to actually run
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
