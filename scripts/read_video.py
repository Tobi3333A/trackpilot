import cv2

video_path = "videos/input/people.mp4"

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise FileNotFoundError(f"Could not open the video: {video_path}")

while True:
    success, frame = video.read()

    if not success:
        break

    cv2.imshow("Trackpilot", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

if not success:
    raise RuntimeError("The video opened but could not be read")

print(f"Frame shape: {frame.shape}")

video.release()
cv2.destroyAllWindows