
import cv2
from deepface import DeepFace

def detect_emotion(frame):
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend="opencv"
        )
        return result[0]['dominant_emotion']
    except:
        return "unknown"

# Start webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break

    # Detect emotion
    emotion = detect_emotion(frame)

    # Display emotion on screen
    label = f"Detected Emotion: {emotion.capitalize()}"
    cv2.putText(frame, label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 175, 25), 2)

    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion()

