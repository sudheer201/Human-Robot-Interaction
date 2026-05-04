import cv2
import numpy as

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    'haarcascade_eye.xml')


def get_eye_position(eye_img):
    # Blur for noise reduction
    eye = cv2.GaussianBlur(eye_img, (7, 7), 0)

    # Threshold (adaptive is better than fixed)
    _, thresh = cv2.threshold(eye, 30, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape

    left = thresh[:, :w//2]
    right = thresh[:, w//2:]

    left_half = cv2.countNonZero(left)
    right_half = cv2.countNonZero(right)

    return left_half, right_half


def detect_gaze(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No Face", "Look at the camera"

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        eyes = eye_detector.detectMultiScale(face_img)

        if len(eyes) < 2:
            return "No Eye Contact", "Look at the camera"

        directions = []

        for (ex, ey, ew, eh) in eyes[:2]:  # use both eyes
            eye = face_img[ey:ey+eh, ex:ex+ew]

            left_half, right_half = get_eye_position(eye)

            if left_half > right_half + 15:
                directions.append("Right")
            elif right_half > left_half + 15:
                directions.append("Left")
            else:
                directions.append("Center")

        # Combine both eyes result
        if directions.count("Left") >= 1:
            return "Looking Left", "Not interested"
        elif directions.count("Right") >= 1:
            return "Looking Right", "Not interested"
        else:
            return "Looking Center", "User engaged"

    return "Unknown", "Unknown"

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        success, frame = camera.read()
        if not success:
            break

        gaze, msg = detect_gaze(frame)

        cv2.putText(frame, gaze, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, msg, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Gaze Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
