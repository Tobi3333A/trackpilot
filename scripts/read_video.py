import cv2
import time
from trackpilot.pipeline import Pipeline
from trackpilot.metrics import Metrics

def main():
    video_path = "videos/input/people.mov"

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise FileNotFoundError(f"Could not open the video: {video_path}")

    source_fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Source FPS: {source_fps:.2f}")

    pipeline = Pipeline("yolo11n.pt")
    metrics = Metrics()

    while True:
        frame_start = time.perf_counter()

        success, frame = video.read()

        if not success:
            break

        frame = pipeline.process_frame(frame, metrics)

        cv2.imshow("Trackpilot", frame)
        key = cv2.waitKey(1)

        frame_end = time.perf_counter()
        pipeline_time = frame_end - frame_start
        metrics.record_pipeline_time(pipeline_time)
        

        if key == ord('q'):
            break

    metrics.print_summary()

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
