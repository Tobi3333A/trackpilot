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

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.25, verbose=False)

    result = results[0]

    for box in result.boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])

        coordinates = box.xyxy[0].tolist()
        confidence = float(box.conf[0])

        label = f"ID {track_id} | {confidence:.2f}"
        x1, y1, x2, y2 = map(int, coordinates)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Trackpilot", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows