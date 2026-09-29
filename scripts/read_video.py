import cv2

video_path = "videos/input/people.mp4"

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise FileNotFoundError(f"Could not open the video: {video_path}")

while True:
    success, frame = video.read()

    if not success:
        break

    height, width, channels = frame.shape
    print(f"Width: {width}, Height: {height},  Channels: {channels}")

    x1 = width // 4
    y1 = height // 4

    x2 = 3 * width // 4
    y2 = 3 * height // 4

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Trackpilot", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows