# app.py
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from transformers import pipeline
import speech_recognition as sr
import tempfile
import re
import os

# Load emotion analysis model once
emotion_classifier = pipeline(
    'text-classification',
    model='j-hartmann/emotion-english-distilroberta-base',
    return_all_scores=True
)

app = Flask(__name__)

# Depression and stress keyword sets
DEPRESSION_KEYWORDS = {
    'high_risk': ['suicide', 'kill myself', 'end it all', 'want to die', 'better off dead', 'harm myself', 'no point living'],
    'medium_risk': ['hopeless', 'empty', 'numb', 'cant go on', 'cant cope', 'overwhelmed', 'dont care anymore'],
    'low_risk': ['sad', 'unhappy', 'down', 'tired', 'sleepy', 'lonely', 'alone']
}

STRESS_KEYWORDS = {
    'high': ['overwhelmed', 'cant handle', 'breaking down', 'too much pressure', 'drowning', 'panic'],
    'medium': ['stressed', 'anxious', 'worried', 'pressure', 'nervous', 'tense'],
    'low': ['busy', 'tired', 'concerned', 'apprehensive']
}

# --- Analysis Functions ---
def analyze_depression_level(text):
    text_lower = text.lower()
    high = sum(1 for k in DEPRESSION_KEYWORDS['high_risk'] if k in text_lower)
    med = sum(1 for k in DEPRESSION_KEYWORDS['medium_risk'] if k in text_lower)
    low = sum(1 for k in DEPRESSION_KEYWORDS['low_risk'] if k in text_lower)

    if high > 0:
        level, score = "DANGEROUS LEVEL", 90
        msg = "⚠️ IMMEDIATE ATTENTION NEEDED: Severe depression signs detected."
    elif med >= 2:
        level, score = "HIGH RISK", 70
        msg = "🔴 High depression risk detected. Professional help is strongly advised."
    elif low >= 3 or med >= 1:
        level, score = "MODERATE RISK", 50
        msg = "🟡 Moderate signs of low mood or sadness."
    else:
        level, score = "LOW RISK", 20
        msg = "🟢 Your text shows no major signs of depression."

    return {'level': level, 'score': score, 'explanation': msg}

def analyze_stress_level(text):
    text_lower = text.lower()
    high = sum(1 for k in STRESS_KEYWORDS['high'] if k in text_lower)
    med = sum(1 for k in STRESS_KEYWORDS['medium'] if k in text_lower)
    low = sum(1 for k in STRESS_KEYWORDS['low'] if k in text_lower)

    polarity = TextBlob(text).sentiment.polarity
    base_stress = abs(polarity) * 30
    keyword_stress = (high * 40) + (med * 25) + (low * 10)
    stress_score = min(100, base_stress + keyword_stress)

    if stress_score >= 70:
        level, msg = "HIGH STRESS", "🔴 High stress detected. Take immediate steps to relax."
    elif stress_score >= 40:
        level, msg = "MODERATE STRESS", "🟡 Moderate stress. Try mindfulness or rest."
    else:
        level, msg = "LOW STRESS", "🟢 You appear calm and balanced."

    return {'level': level, 'score': int(stress_score), 'explanation': msg}

def analyze_emotions(text):
    results = emotion_classifier(text)[0]
    emotions = {r['label']: round(r['score'] * 100, 1) for r in results}
    dominant = max(emotions, key=emotions.get)
    return {'dominant': dominant, 'all_emotions': emotions}

def get_recommendations(depression, stress):
    rec = []

    if depression['score'] >= 70:
        rec += [
            "🚨 Contact a mental health professional immediately.",
            "Call Helpline: 988 (US) or your local support line.",
            "Avoid isolation—reach out to trusted people."
        ]
    elif depression['score'] >= 50:
        rec += [
            "📞 Talk to a counselor or therapist soon.",
            "Keep a daily gratitude or mood journal.",
            "Focus on sleep, sunlight, and healthy meals."
        ]
    else:
        rec += [
            "💚 Keep practicing self-care.",
            "Maintain social connections.",
            "Stay active and hydrated."
        ]

    if stress['score'] >= 70:
        rec += [
            "🧘 Try breathing exercises (4-7-8 technique).",
            "Take breaks often and delegate tasks."
        ]
    elif stress['score'] >= 40:
        rec += [
            "🎧 Listen to calming music.",
            "Take short walks or stretch breaks."
        ]

    return rec

# --- Voice Recognition ---
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text_input = None

    # Case 1: Text input
    if 'text' in request.form and request.form['text'].strip():
        text_input = request.form['text'].strip()

    # Case 2: Voice input (file)
    elif 'audio' in request.files:
        file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            file.save(temp_audio.name)
            text_input = speech_to_text(temp_audio.name)
        os.remove(temp_audio.name)

    if not text_input:
        return jsonify({'error': 'No valid text or voice input detected'})

    # Run analysis
    depression = analyze_depression_level(text_input)
    stress = analyze_stress_level(text_input)
    emotion = analyze_emotions(text_input)
    recs = get_recommendations(depression, stress)

    return jsonify({
        'text': text_input,
        'depression': depression,
        'stress': stress,
        'emotion': emotion,
        'recommendations': recs
    })

if __name__ == '__main__':
    app.run(debug=True)
