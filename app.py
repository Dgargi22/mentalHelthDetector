rom flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
from transformers import pipeline
import speech_recognition as sr
import io
import os

# --- Load emotion model ---
print("🔄 Loading emotion classification model...")
emotion_classifier = pipeline(
    'text-classification',
    model='j-hartmann/emotion-english-distilroberta-base',
    top_k=None
)
print("✅ Emotion model loaded successfully.")

# --- Flask setup ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- Keyword Dictionaries ---
DEPRESSION_KEYWORDS = {
    'high_risk': ['suicide', 'kill myself', 'end it all', 'want to die', 'better off dead', 'harm myself'],
    'medium_risk': ['hopeless', 'empty', 'numb', 'cant go on', 'worthless', 'no future'],
    'low_risk': ['sad', 'unhappy', 'down', 'crying', 'exhausted', 'lonely']
}

STRESS_KEYWORDS = {
    'high': ['overwhelmed', 'cant handle', 'breaking down', 'too much pressure', 'panic', 'panic attack'],
    'medium': ['stressed', 'anxious', 'worried', 'pressure', 'tense', "can't sleep"],
    'low': ['busy', 'tired', 'concerned', 'distracted']
}

# --- Emotion Analysis ---
def analyze_emotions(text):
    try:
        truncated = text[:1000]
        results = emotion_classifier(truncated)[0]
        emotions = {r['label']: round(r['score'] * 100, 1) for r in results}
        dominant = max(emotions, key=emotions.get)
        return {'dominant': dominant, 'all_emotions': emotions}
    except Exception as e:
        print(f"[Error: Emotion Analysis] {e}")
        return {'dominant': 'neutral', 'all_emotions': {'neutral': 100.0}}

# --- Depression Analysis ---
def analyze_depression_level(text, emotions):
    text = text.lower()
    high = sum(k in text for k in DEPRESSION_KEYWORDS['high_risk'])
    med = sum(k in text for k in DEPRESSION_KEYWORDS['medium_risk'])
    low = sum(k in text for k in DEPRESSION_KEYWORDS['low_risk'])
    keyword_score = (high * 90) + (med * 35) + (low * 10)

    multiplier = 1.0
    if emotions.get('dominant') in ['joy', 'love']:
        multiplier = 0.3
    elif emotions.get('dominant') in ['sadness', 'fear']:
        multiplier = 1.2

    final_score = min(100, keyword_score * multiplier)

    if final_score >= 85:
        return {'level': 'DANGEROUS LEVEL', 'score': 95,
                'explanation': "⚠️ Severe distress detected. Please talk to someone immediately."}
    elif final_score >= 65:
        return {'level': 'HIGH RISK', 'score': 75,
                'explanation': "🔴 High depression risk detected. Professional help is advised."}
    elif final_score >= 35:
        return {'level': 'MODERATE RISK', 'score': 50,
                'explanation': "🟡 Some sadness detected. Take care and reach out if needed."}
    elif final_score > 10:
        return {'level': 'LOW RISK', 'score': 20,
                'explanation': "🟢 Mild sadness signs. Stay mindful."}
    else:
        return {'level': 'VERY LOW', 'score': 10,
                'explanation': "🟢 No depressive cues detected."}

# --- Stress Analysis ---
def analyze_stress_level(text, emotions):
    text = text.lower()
    high = sum(k in text for k in STRESS_KEYWORDS['high'])
    med = sum(k in text for k in STRESS_KEYWORDS['medium'])
    low = sum(k in text for k in STRESS_KEYWORDS['low'])

    polarity = TextBlob(text).sentiment.polarity
    base = (1 - polarity) * 20
    keyword_stress = (high * 40) + (med * 25) + (low * 10)
    total = base + keyword_stress

    multiplier = 1.0
    if emotions.get('dominant') in ['joy', 'love']:
        multiplier = 0.3
    elif emotions.get('dominant') in ['anger', 'fear']:
        multiplier = 1.3

    score = min(100, total * multiplier)

    if score >= 70:
        return {'level': 'HIGH STRESS', 'score': int(score),
                'explanation': "🔴 High stress detected. Try to pause and calm your mind."}
    elif score >= 40:
        return {'level': 'MODERATE STRESS', 'score': int(score),
                'explanation': "🟡 Moderate tension detected. Relaxation may help."}
    else:
        return {'level': 'LOW STRESS', 'score': int(score),
                'explanation': "🟢 You appear calm and balanced."}

# --- Recommendations ---
def get_recommendations(depression, stress):
    recs = []
    if depression['score'] >= 85:
        recs.append("Reach out to a trusted friend or helpline immediately.")
    if stress['score'] >= 70:
        recs.append("Try breathing deeply and take short breaks.")
    if depression['score'] >= 35 and stress['score'] >= 40:
        recs.append("Journaling your thoughts can help bring clarity.")
    if depression['score'] < 35 and stress['score'] < 40:
        recs.append("Keep maintaining your mental balance — great job!")
    return list(dict.fromkeys(recs))[:3]

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text_input = None

        # --- Handle text input ---
        if 'text' in request.form and request.form['text'].strip():
            text_input = request.form['text'].strip()

        # --- Handle audio input ---
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(audio_file.read())) as source:
                audio_data = recognizer.record(source)
                try:
                    text_input = recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    return jsonify({'error': 'Could not understand audio.'})
                except sr.RequestError:
                    return jsonify({'error': 'Speech recognition service unavailable.'})

        if not text_input:
            return jsonify({'error': 'No valid text or audio input received.'})

        emotion = analyze_emotions(text_input)
        depression = analyze_depression_level(text_input, emotion)
        stress = analyze_stress_level(text_input, emotion)
        recs = get_recommendations(depression, stress)

        return jsonify({
            'text': text_input,
            'emotion': emotion,
            'depression': depression,
            'stress': stress,
            'recommendations': recs
        })

    except Exception as e:
        print(f"[Error: /analyze] {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'})

if __name__ == '__main__':
    print("🚀 Flask server starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
