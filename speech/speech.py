import sounddevice as sd
import numpy as np
import whisper
from scipy.io.wavfile import write
from textblob import TextBlob

# Load Whisper model
model = whisper.load_model("tiny")

fs = 16000
duration = 10

def record_audio():
    print("Listening...")

    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.05:
        return "positive"
    elif polarity < -0.05:
        return "negative"
    else:
        return "neutral"
    return sentiment, polarity

def speech_to_text():
    audio = record_audio()

    result = model.transcribe(audio)
    text = result["text"].strip()

    print("Transcription:", result["text"])
    sentiment, polarity = analyze_sentiment(text)
    
    print("Sentiment:", text)
    print("Sentiment:", sentiment)
    print("Polarity:", polarity)
    return text, sentiment, polarity

if __name__ == "__main__":
    speech_to_text()
