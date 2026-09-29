import cv2
from ultralytics import YOLO

video_path = "videos/input/people.mov"

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise FileNotFoundError(f"Could not open the video: {video_path}")

model = YOLO("yolo11n.pt")

while True:
    success, frame = video.read()

    if not success:
        break

    results = model.predict(frame, classes=[0], conf=0.4, verbose=False)

    result = results[0]

    for box in result.boxes:
        coordinates = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, coordinates)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Person: {confidence:.2f}", (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Trackpilot", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows