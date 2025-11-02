from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
from transformers import pipeline
import speech_recognition as sr
import tempfile
import re
import os

# --- Model & App Setup ---

print("Loading emotion classification model...")
emotion_classifier = pipeline(
    'text-classification',
    model='j-hartmann/emotion-english-distilroberta-base',
    top_k=None 
)
print("Emotion model loaded successfully.")

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app) 

# --- Keyword Definitions ---
DEPRESSION_KEYWORDS = {
    'high_risk': ['suicide', 'kill myself', 'end it all', 'want to die', 'better off dead', 'harm myself', 'no point living'],
    'medium_risk': ['hopeless', 'empty', 'numb', 'cant go on', 'cant cope', 'worthless', 'no future'],
    'low_risk': ['sad', 'unhappy', 'down', 'crying', 'exhausted', 'sleepy', 'lonely', 'alone']
}

STRESS_KEYWORDS = {
    'high': ['overwhelmed', 'cant handle', 'breaking down', 'too much pressure', 'drowning', 'panic', 'panic attack'],
    'medium': ['stressed', 'anxious', 'worried', 'pressure', 'nervous', 'tense', 'can\'t sleep'],
    'low': ['busy', 'tired', 'concerned', 'apprehensive', 'distracted']
}

# --- Analysis Functions ---

def analyze_depression_level(text, emotions):
    text_lower = text.lower()
    high = sum(1 for k in DEPRESSION_KEYWORDS['high_risk'] if k in text_lower)
    med = sum(1 for k in DEPRESSION_KEYWORDS['medium_risk'] if k in text_lower)
    low = sum(1 for k in DEPRESSION_KEYWORDS['low_risk'] if k in text_lower)
    keyword_score = (high * 90) + (med * 35) + (low * 10)
    
    dominant_emotion = emotions.get('dominant', 'neutral')
    emotion_multiplier = 1.0

    if dominant_emotion in ['joy', 'love', 'optimism']:
        emotion_multiplier = 0.2
    elif dominant_emotion in ['sadness', 'fear', 'disgust']:
        emotion_multiplier = 1.2
    
    final_score = min(100, keyword_score * emotion_multiplier)
    
    if final_score >= 85:
        level, score = "DANGEROUS LEVEL", 95
        msg = "⚠️ Your words show significant signs of distress. Please connect with someone immediately. You can call or text 988 (US) or your local crisis line."
    elif final_score >= 65:
        level, score = "HIGH RISK", 75
        msg = "🔴 High depression risk detected. These are heavy feelings. Professional help is strongly advised."
    elif final_score >= 35:
        level, score = "MODERATE RISK", 50
        msg = "🟡 We're noticing moderate signs of low mood. It's important to be gentle with yourself today."
    elif final_score > 10:
        level, score = "LOW RISK", 20
        msg = "🟢 Your text shows no major signs of depression. Good to see you checking in."
    else:
        level, score = "VERY LOW", 10
        msg = "🟢 Your text shows no major signs of depression. We're glad to see that."

    return {'level': level, 'score': int(final_score), 'explanation': msg}

def analyze_stress_level(text, emotions):
    text_lower = text.lower()
    high = sum(1 for k in STRESS_KEYWORDS['high'] if k in text_lower)
    med = sum(1 for k in STRESS_KEYWORDS['medium'] if k in text_lower)
    low = sum(1 for k in STRESS_KEYWORDS['low'] if k in text_lower)

    polarity = TextBlob(text).sentiment.polarity 
    base_stress = (1 - polarity) * 20
    keyword_stress = (high * 40) + (med * 25) + (low * 10)
    keyword_score = base_stress + keyword_stress
    
    dominant_emotion = emotions.get('dominant', 'neutral')
    emotion_multiplier = 1.0

    if dominant_emotion in ['joy', 'love', 'calm']:
        emotion_multiplier = 0.3
    elif dominant_emotion in ['anger', 'anxiety', 'fear']:
        emotion_multiplier = 1.3
        
    final_score = min(100, keyword_score * emotion_multiplier)

    if final_score >= 70:
        level, msg = "HIGH STRESS", "🔴 High stress detected. You seem overwhelmed. Take immediate steps to find a moment of calm."
    elif final_score >= 40:
        level, msg = "MODERATE STRESS", "🟡 Moderate stress. We're noticing tension. Try mindfulness or rest."
    else:
        level, msg = "LOW STRESS", "🟢 You appear calm and balanced. Keep up the healthy habits."

    return {'level': level, 'score': int(final_score), 'explanation': msg}

def analyze_emotions(text):
    try:
        # Truncate text for model stability
        truncated_text = text[:1000]
        results = emotion_classifier(truncated_text)[0]
        emotions = {r['label']: round(r['score'] * 100, 1) for r in results}
        dominant = max(emotions, key=emotions.get)
        return {'dominant': dominant, 'all_emotions': emotions}
    except Exception as e:
        print(f"Error in emotion analysis (text might be too short/unusual): {e}")
        return {'dominant': 'neutral', 'all_emotions': {'neutral': 100.0}}

def get_recommendations(depression, stress):
    recs = []
    
    if depression['score'] >= 85:
        recs.append("Please connect with someone immediately. You can call or text 988 (US) or your local crisis line. They are there to listen.")
    if stress['score'] >= 70:
        recs.append("You seem overwhelmed. Try a 5-4-3-2-1 grounding exercise: Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.")
    if depression['score'] >= 65 and depression['score'] < 85:
        recs.append("It sounds like you're carrying a heavy load. It might be a good time to talk to a therapist or a trusted friend about these feelings.")
    if stress['score'] >= 40 and stress['score'] < 70:
        recs.append("Your tension levels seem elevated. A short 10-minute walk outside, without your phone, can make a big difference.")
    if depression['score'] < 35 and stress['score'] < 40:
        recs.append("It's great that you're checking in. Keep up the self-awareness! Maintaining this balance is a healthy practice.")
    if not recs:
        recs.append("Taking a moment to check in with yourself is a healthy step. Continue to be mindful of your feelings as you go about your day.")

    return list(dict.fromkeys(recs))[:3]

# --- Voice Recognition ---
def speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
        print("Transcribing audio...")
        text = recognizer.recognize_google(audio_data)
        print(f"Transcription: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        # THIS IS THE ERROR YOU WERE SEEING
        print(f"Error in speech_to_text: {e}") 
        return None

# --- API Routes ---

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/analyze', methods=['POST'])
def analyze():
    text_input = None
    temp_webm_path = None # Original file from browser
    temp_wav_path = None  # Converted file for speech_recognition
    print("Received request to /analyze")

    try:
        if 'text' in request.form and request.form['text'].strip():
            text_input = request.form['text'].strip()
            print(f"Received text input: {text_input[:50]}...")
        elif 'audio' in request.files:
            print("Received audio file.")
            file = request.files['audio']
            
            # 1. Save the original .webm file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                file.save(temp_audio.name)
                temp_webm_path = temp_audio.name

            # 2. Create a name for the output .wav file
            temp_wav_path = temp_webm_path.replace(".webm", ".wav")

            # 3. Run ffmpeg to convert the file
            print(f"Converting {temp_webm_path} to {temp_wav_path}...")
            # -loglevel quiet: supresses ffmpeg's console spam
            # -y: overwrite output file if it exists
            # -ac 1: set audio channels to 1 (mono)
            # -ar 16000: set audio sample rate to 16kHz (good for STT)
            command = f"ffmpeg -i {temp_webm_path} -ac 1 -ar 16000 -y {temp_wav_path} -loglevel quiet"
            return_code = os.system(command) # Run the command
            
            if return_code != 0:
                print(f"ffmpeg conversion failed with code {return_code}")
                raise Exception("ffmpeg audio conversion failed.") 
            
            # 4. Pass the *converted* .wav file to speech_to_text
            text_input = speech_to_text(temp_wav_path)
            
        if not text_input:
            print("Analysis failed: No valid text or unclear audio.")
            # *** FIX: Return a 400 Bad Request error to trigger JS .catch() ***
            return jsonify({'error': 'No valid text or voice input detected, or audio was unclear.'}), 400

        print("Running AI analysis...")
        emotion = analyze_emotions(text_input) 
        depression = analyze_depression_level(text_input, emotion) 
        stress = analyze_stress_level(text_input, emotion) 
        recs = get_recommendations(depression, stress)
        print("Analysis complete. Sending response.")

        return jsonify({
            'text': text_input,
            'depression': depression,
            'stress': stress,
            'emotion': emotion,
            'recommendations': recs
        })

    except Exception as e:
        print(f"!!! FATAL ANALYSIS ERROR: {e}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500
    
    finally:
        # 5. Clean up *both* files
        if temp_webm_path and os.path.exists(temp_webm_path):
            os.remove(temp_webm_path)
            print(f"Cleaned up temp file: {temp_webm_path}")
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            print(f"Cleaned up temp file: {temp_wav_path}")

# --- Main Run Block ---
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)


