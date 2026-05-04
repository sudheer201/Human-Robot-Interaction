
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break

    # pose estimation
    results = model(frame, verbose=False)
    
    if initial_speed is None:
        speed = results[0].speed
        initial_speed = (speed["preprocess"] , speed["inference"] , speed["postprocess"]) 

  
    output_frame = results[0].plot()

    cv2.putText(output_frame, f"Preprocess: {results[0].speed['preprocess']:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("pose Detection", output_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
