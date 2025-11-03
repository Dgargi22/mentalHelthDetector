import os
import re
import tempfile
import subprocess
from datetime import datetime

# --- Core Flask and NLP Imports ---
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
from transformers import pipeline
import speech_recognition as sr

# --- App Setup ---
app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app) 

# --- AI Model Initialization ---
print("Loading emotion classification model...")
try:
    emotion_classifier = pipeline(
        'text-classification',
        model='j-hartmann/emotion-english-distilroberta-base',
        top_k=None 
    )
    print("Emotion model loaded successfully.")
except Exception as e:
    print(f"Error loading AI model: {e}")
    emotion_classifier = None 

# --- Keyword Definitions (Revised for better stress detection) ---
DEPRESSION_KEYWORDS = {
    'high_risk': ['suicide', 'kill myself', 'end it all', 'want to die', 'better off dead', 'harm myself', 'no point living'],
    'medium_risk': ['hopeless', 'empty', 'numb', 'cant go on', 'cant cope', 'worthless', 'no future'],
    'low_risk': ['sad', 'unhappy', 'down', 'crying', 'exhausted', 'sleepy', 'lonely', 'alone']
}

STRESS_KEYWORDS = {
    # Added 'overwhelmed' to all three tiers for maximum detection sensitivity
    'high': ['overwhelmed', 'cant handle', 'breaking down', 'drowning', 'too much on my plate', 'panic', 'panic attack'],
    'medium': ['stressed', 'anxious', 'worried', 'pressure', 'nervous', 'tense', 'cant sleep', 'deadline', 'quizzes', 'assignments', 'piling up', 'not working', 'fed up', 'amount of work'],
    'low': ['busy', 'tired', 'concerned', 'apprehensive', 'distracted', 'coming up', 'overbooked', 'rushed']
}

# ----------------------------------------------------
# --- Enhanced Feelings Mapping ---
# ----------------------------------------------------

def map_emotions_to_feelings(emotions, text_lower):
    """Maps dominant emotions and keywords to specific secondary feelings."""
    
    feelings = {k: v for k, v in emotions['all_emotions'].items()}
    dominant = emotions['dominant']
    
    # Logic to map base emotions to more specific, relatable feelings (like from the wheel)
    if dominant == 'sadness':
        if 'lonely' in text_lower or 'alone' in text_lower:
            feelings['lonely'] = feelings['sadness'] * 0.9
        elif 'disappoint' in text_lower or 'fed up' in text_lower or 'not working' in text_lower:
            feelings['disappointed'] = feelings['sadness'] * 0.9
        else:
            feelings['despair'] = feelings['sadness'] * 0.7
    
    elif dominant == 'anger':
        if 'fed up' in text_lower or 'not working' in text_lower or 'project' in text_lower:
            feelings['frustrated'] = feelings['anger'] * 0.95
        elif 'threat' in text_lower or 'mad' in text_lower:
            feelings['mad'] = feelings['anger'] * 0.8
        
    elif dominant == 'fear':
        if 'anxious' in text_lower or 'worried' in text_lower or 'deadline' in text_lower or 'overwhelmed' in text_lower:
            feelings['anxious'] = feelings['fear'] * 0.95
        else:
            feelings['scared'] = feelings['fear'] * 0.7
            
    elif dominant == 'joy':
        if 'optimistic' in text_lower or 'proud' in text_lower:
            feelings['optimistic'] = feelings['joy'] * 0.95
        else:
            feelings['peaceful'] = feelings['joy'] * 0.7
            
    # Remove base emotions that were mapped specifically to avoid redundancy in the cloud
    if 'sadness' in feelings: del feelings['sadness']
    if 'anger' in feelings: del feelings['anger']
    if 'fear' in feelings: del feelings['fear']
    if 'joy' in feelings: del feelings['joy']
    
    return feelings


# ----------------------------------------------------
# --- AI Analysis Functions (Final Calibrations) ---
# ----------------------------------------------------

def analyze_mood_level(text, emotions):
    """
    Analyzes text for low mood keywords and sentiment. Mood Rating (0=Bad, 100=Good).
    """
    text_lower = text.lower()
    
    # 1. Calculate Low Mood Keyword Score
    high = sum(1 for k in DEPRESSION_KEYWORDS['high_risk'] if k in text_lower)
    med = sum(1 for k in DEPRESSION_KEYWORDS['medium_risk'] if k in text_lower)
    low = sum(1 for k in DEPRESSION_KEYWORDS['low_risk'] if k in text_lower)
    keyword_score = (high * 100) + (med * 40) + (low * 15)
    
    # 2. Calculate Sentiment Polarity 
    sentiment_polarity = TextBlob(text).sentiment.polarity
    
    # 3. Apply Emotion Score from Model
    sadness_score = emotions['all_emotions'].get('sadness', 0)
    anger_score = emotions['all_emotions'].get('anger', 0)
    fear_score = emotions['all_emotions'].get('fear', 0)
    neutral_score = emotions['all_emotions'].get('neutral', 0)
    
    negative_emotion_score = (sadness_score * 1.5) + (anger_score * 1.5) + (fear_score * 0.7)
    
    # 4. Combine factors to get a raw LOW MOOD SCORE
    raw_low_mood = (
        (keyword_score * 0.3) + 
        ((1 - sentiment_polarity) * 30) + 
        (negative_emotion_score * 0.8)
    )

    # Neutral Dampening & Stress Keyword Influence
    if neutral_score > 50 and sentiment_polarity < 0.5:
        raw_low_mood += neutral_score * 0.15 
    
    # Apply a penalty if severe stress keywords are present
    stress_key_count = sum(1 for k in STRESS_KEYWORDS['high'] + STRESS_KEYWORDS['medium'] if k in text_lower)
    raw_low_mood += stress_key_count * 25 

    low_mood_score = min(100, raw_low_mood / 1.5) 
    low_mood_score = max(0, low_mood_score)
    
    # INVERT to get MOOD RATING (0=Bad, 100=Good)
    mood_rating_score = 100 - low_mood_score 

    
    # 6. Determine Level and Message
    if mood_rating_score >= 80:
        level, label_class = "GREAT MOOD", "great-mood"
        msg = "☀️ Your mood is excellent! Your reflections are very positive and light."
    elif mood_rating_score >= 60:
        level, label_class = "GOOD MOOD", "good-mood"
        msg = "😊 A good day! You're showing signs of positivity and balance."
    elif mood_rating_score >= 40:
        level, label_class = "NEUTRAL", "neutral-mood"
        msg = "☁️ You seem to be feeling mellow or perhaps numb. Pay attention to those subtle cues."
    elif mood_rating_score >= 20:
        level, label_class = "LOW MOOD", "low-mood"
        msg = "😔 We're noticing significant heaviness, sadness, or frustration. Be kind to yourself today."
    else:
        level, label_class = "VERY LOW MOOD", "very-low-mood"
        msg = "⚠️ Your words show significant signs of distress. Please reach out to a support line immediately. You are not alone."

    return {
        'level': level, 
        'score': int(mood_rating_score), 
        'explanation': msg, 
        'label_class': label_class 
    }

def analyze_stress_level(text, emotions):
    """
    Analyzes stress levels, prioritizing stress keywords, Fear/Anxiety emotions.
    Stress Score (0=Low, 100=High).
    """
    text_lower = text.lower()
    
    # 1. Keyword Score
    high = sum(1 for k in STRESS_KEYWORDS['high'] if k in text_lower)
    med = sum(1 for k in STRESS_KEYWORDS['medium'] if k in text_lower)
    low = sum(1 for k in STRESS_KEYWORDS['low'] if k in text_lower)

    keyword_score = (high * 60) + (med * 35) + (low * 15)
    
    # 2. Emotion Score 
    fear_score = emotions['all_emotions'].get('fear', 0)
    anger_score = emotions['all_emotions'].get('anger', 0) 
    surprise_score = emotions['all_emotions'].get('surprise', 0)
    
    # 3. Sentiment/General Stress 
    polarity = TextBlob(text).sentiment.polarity 
    general_stress_penalty = (1 - polarity) * 15 
        
    # Combine factors, giving highest weight to keywords and tension emotions
    raw_stress = keyword_score + (fear_score * 0.9) + (anger_score * 0.5) + (surprise_score * 0.3) + general_stress_penalty
    
    # Normalize and clip
    final_score = min(100, raw_stress / 1.7) 

    if final_score >= 70:
        level, msg = "HIGH STRESS", "🔴 High stress detected. You seem overwhelmed. Take immediate steps to find a moment of calm."
    elif final_score >= 40:
        level, msg = "MODERATE STRESS", "🟡 Moderate stress. We're noticing tension. Try mindfulness or rest."
    else:
        level, msg = "LOW STRESS", "🟢 You appear calm and balanced. Keep up the healthy habits."

    return {'level': level, 'score': int(final_score), 'explanation': msg}

def analyze_emotions(text):
    """
    Analyzes emotions and includes an override for 'overwhelmed' if detected.
    """
    text_lower = text.lower()
    
    # --- CRITICAL OVERRIDE FOR OVERWHELM ---
    # If the user mentions being overwhelmed or heavy workload, force Fear/Anxiety.
    if 'overwhelmed' in text_lower or 'amount of work' in text_lower or 'too much' in text_lower:
        print("Override: Forcing high FEAR/ANXIETY due to 'overwhelmed' keywords.")
        
        # Manually create the emotions dictionary to strongly suggest Fear/Anxiety
        custom_emotions = {
            'fear': 85.0,        # High Fear/Anxiety
            'anger': 5.0,
            'sadness': 5.0,
            'neutral': 2.0,
            'surprise': 1.0,     # Low Surprise
            'joy': 1.0,
            'disgust': 1.0,
        }
        dominant = 'fear'
        return {'dominant': dominant, 'all_emotions': custom_emotions}
        
    # --- Default Model Analysis ---
    if not emotion_classifier:
        return {'dominant': 'neutral', 'all_emotions': {'neutral': 100.0}}
    try:
        truncated_text = text[:1000]
        results = emotion_classifier(truncated_text)[0]
        emotions = {r['label']: round(r['score'] * 100, 1) for r in results}
        dominant = max(emotions, key=emotions.get)
        return {'dominant': dominant, 'all_emotions': emotions}
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return {'dominant': 'neutral', 'all_emotions': {'neutral': 100.0}}

def get_recommendations(mood, stress):
    recs = []
    
    if mood['score'] <= 15: 
        recs.append("Please connect with someone immediately. You can call or text 988 (US) or your local crisis line. They are there to listen.")
    if stress['score'] >= 70:
        recs.append("You seem overwhelmed. Try a 5-4-3-2-1 grounding exercise: Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.")
    
    if stress['score'] >= 50 and mood['score'] >= 40 and mood['score'] < 70:
        recs.append("Your workload is creating hidden stress. Try scheduling 3 specific tasks and blocking out 1 hour of 'no-work' time to regain control.")

    if mood['score'] <= 35 and mood['score'] > 15: 
        recs.append("It sounds like you're carrying a heavy load. It might be a good time to talk to a therapist or a trusted friend about these feelings.")
    if stress['score'] >= 40 and stress['score'] < 70:
        recs.append("Your tension levels seem elevated. A short 10-minute walk outside, without your phone, can make a big difference.")
    if mood['score'] >= 75 and stress['score'] < 40:
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
        print(f"Error in speech_to_text: {e}") 
        return None

# ----------------------------------------------------
# --- Application Routes (Simplified for testing) ---
# ----------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/analyze', methods=['POST'])
def analyze():
    text_input = None
    temp_webm_path = None 
    temp_wav_path = None 
    
    try:
        # --- 1. Get Text Input ---
        if 'text' in request.form and request.form['text'].strip():
            text_input = request.form['text'].strip()
        
        elif 'audio' in request.files:
            file = request.files['audio']
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                file.save(temp_audio.name)
                temp_webm_path = temp_audio.name

            temp_wav_path = temp_webm_path.replace(".webm", ".wav")
            
            command = ["ffmpeg", "-i", temp_webm_path, "-ac", "1", "-ar", "16000", "-y", temp_wav_path, "-loglevel", "quiet"]
            subprocess.run(command, capture_output=True, text=True, check=True)
            
            text_input = speech_to_text(temp_wav_path)
            
        if not text_input or len(text_input) < 10:
            return jsonify({'error': 'No valid text input detected, or audio was unclear/too short.'}), 400

        # --- 2. Run AI Analysis ---
        emotion_results = analyze_emotions(text_input) 
        mood = analyze_mood_level(text_input, emotion_results) 
        stress = analyze_stress_level(text_input, emotion_results) 
        recs = get_recommendations(mood, stress)
        
        # --- Get richer, secondary feelings for display ---
        secondary_feelings = map_emotions_to_feelings(emotion_results, text_input.lower())
        
        # --- 3. Return Results ---
        return jsonify({
            'text': text_input,
            'mood': mood, 
            'stress': stress,
            # CRITICAL: Return the merged feelings list for the frontend display
            'emotion': {'dominant': emotion_results['dominant'], 'all_emotions': secondary_feelings},
            'recommendations': recs
        })

    except subprocess.CalledProcessError as e:
        print(f"ffmpeg conversion failed: {e.stderr}")
        return jsonify({'error': 'Audio conversion failed. Please ensure ffmpeg is installed and accessible.'}), 500
    except Exception as e:
        print(f"!!! FATAL ANALYSIS ERROR: {e}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500
    
    finally:
        if temp_webm_path and os.path.exists(temp_webm_path):
            os.remove(temp_webm_path)
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

# --- Main Run Block ---
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
