
import cv2
import threading
import sounddevice as sd
import numpy as np
import whisper
from textblob import TextBlob
from deepface import DeepFace
from ultralytics import YOLO

# LOAD MODEL

whisper_model = whisper.load_model("tiny")
pose_model = YOLO("yolov8n-pose.pt")


speech_text = ""
speech_sentiment = ""
emotion_text = ""
gaze_text = ""
engagement_text = ""

# SPEECH + SENTIMENT ANALYSIS

sample_rate = 16000
duration = 10

def record_audio():
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

def speech_loop():
    global speech_text, speech_sentiment

    while True:
        try:
            audio = record_audio()
            result = whisper_model.transcribe(audio)

            speech_text = result["text"]
            speech_sentiment = analyze_sentiment(speech_text)

        except Exception as e:
            speech_text = "Error"
            speech_sentiment = "Error"


# EMOTION DETECTION

def detect_emotion(frame):
    global emotion_text
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend="opencv"
        )
        emotion_text = result[0]['dominant_emotion']
    except:
        emotion_text = "unknown"


# GAZE DETECTION

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    'haarcascade_eye.xml')

def get_eye_position(eye_img):
    eye = cv2.GaussianBlur(eye_img, (7, 7), 0)
    _, thresh = cv2.threshold(eye, 30, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    left = thresh[:, :w//2]
    right = thresh[:, w//2:]

    return cv2.countNonZero(left), cv2.countNonZero(right)

def detect_gaze(frame):
    global gaze_text, engagement_text

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        gaze_text = "No Face"
        engagement_text = "Please Look at camera"
        return

    for (x, y, w, h) in faces:
        sample_gray = gray[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(sample_gray)

        if len(eyes) < 2:
            gaze_text = "No Eye Contact"
            engagement_text = "Look at camera"
            return

        directions = []

        for (ex, ey, ew, eh) in eyes[:2]:
            eye = sample_gray[ey:ey+eh, ex:ex+ew]
            left, right = get_eye_position(eye)

            if left > right + 15:
                directions.append("Right")
            elif right > left + 15:
                directions.append("Left")
            else:
                directions.append("Center")

        if "Left" in directions:
            gaze_text = "Looking Left"
            engagement_text = "Not engaged"
        elif "Right" in directions:
            gaze_text = "Looking Right"
            engagement_text = "Not engaged"
        else:
            gaze_text = "Looking Center"
            engagement_text = "Engaged"


# SPEECH THREADING FOR REAL-TIME AND ALLOWS MULTIPLE PROCESSES TO RUN SIMULTANEOUSLY

threading.Thread(target=speech_loop, daemon=True).start()


camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break

    # Pose detection
    results = pose_model(frame)
    frame = results[0].plot()

    # Emotion + gaze
    detect_emotion(frame)
    detect_gaze(frame)


    # FINAL OUTPUT DISPLAY

    cv2.putText(frame, f"Emotion: {emotion_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    cv2.putText(frame, f"Gaze: {gaze_text}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    cv2.putText(frame, f"Engagement: {engagement_text}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    cv2.putText(frame, f"Speech: {speech_text}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    cv2.putText(frame, f"Sentiment: {speech_sentiment}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    cv2.imshow("AI Interview Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
