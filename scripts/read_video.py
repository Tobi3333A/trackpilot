import cv2
from ultralytics import YOLO
import time

video_path = "videos/input/people.mov"

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise FileNotFoundError(f"Could not open the video: {video_path}")

source_fps = video.get(cv2.CAP_PROP_FPS)
print(f"Source FPS: {source_fps:.2f}")

pipeline_times = []
track_times = []

model = YOLO("yolo11n.pt")

target_id = None

tracked_frames = 0
lost_frames = 0

while True:
    frame_start = time.perf_counter()

    success, frame = video.read()

    if not success:
        break

    target_found = False
    target_center = None
    target_area_ratio = None

    yaw_command = 0.0
    command = "Stop"

    forward_command = 0.0
    distance_command = 'Stop'

    track_time_start = time.perf_counter()
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.25, verbose=False)
    track_time_end = time.perf_counter()
    track_time = track_time_end - track_time_start
    track_times.append(track_time)

    # print(type(results))
    # print(len(results))
    
    result = results[0]

    # print(result)
    # print(result.orig_shape)
    # print(result.names)
    # print(f"Boxes: {result.boxes}\n\n")
    # print(f"Boxes Length: {len(result.boxes)}\n\n")

    for box in result.boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])

        # print(f"Box: {box}")
        # print(f"Box id: {box.id}")
        # print(f"Track id: {track_id}\n\n")

        if target_id is None:
            target_id = track_id

        is_target = track_id == target_id

        coordinates = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coordinates)

        confidence = float(box.conf[0])

        if is_target:
            color = (0, 0, 255)
            label = f"TARGET {track_id} | {confidence:.2f}"
            target_center_x = (x1 + x2) // 2
            target_center_y = (y1 + y2) // 2
            target_center = (target_center_x, target_center_y)
            
            target_width = x2 - x1
            target_height = y2 - y1
            target_area = target_width * target_height

            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_width * frame_height
            target_area_ratio = target_area / frame_area
            
            target_found = True
        else:
            color = (0, 255, 0)
            label = f"ID {track_id} | {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

    if target_id is None:
        state = 'Searching'
    elif target_found:
        state = 'Tracking'
    else:
        state = 'Target Lost'

    state_color = (0, 255, 0) if state == 'Tracking' else (0, 0, 255)

    cv2.putText(frame, f"State: {state}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, state_color, 2)

    if state == 'Tracking':
        tracked_frames += 1

        cv2.putText(frame, f"Target size: {target_area_ratio:.3f}", (20, 110), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)

        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        error_x = target_center_x - frame_center_x
        normalized_error_x = error_x / (frame_width / 2)

        deadband = 0.05
        kp = 0.8

        if abs(normalized_error_x) <= deadband:
            yaw_command = 0.0
            command = 'Hold'
        else:
            yaw_command = kp * normalized_error_x
            yaw_command = max(-1.0, min(1.0, yaw_command))

            if yaw_command < 0:
                command = 'Yaw Left'
            else:
                command = 'Yaw Right'

        desired_area_ratio = 0.12
        distance_deadband = 0.01
        kp_distance = 3.0

        distance_error = desired_area_ratio - target_area_ratio

        if abs(distance_error) <= distance_deadband:
            forward_command = 0.0
            distance_command = 'Hold distance'
        else:
            forward_command = kp_distance * distance_error
            forward_command = max(-1.0, min(1.0, forward_command))

            if forward_command > 0:
                distance_command = 'Move forward'
            else:
                distance_command = 'Move backward'

        cv2.putText(frame, f"Distance: {distance_command} ({forward_command:.2f})", (20, 145), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

        cv2.circle(frame, (frame_center_x, frame_center_y), 6, (255, 0, 0), -1)
        cv2.circle(frame, target_center, 6, (0, 0, 255), -1)
        cv2.line(frame, (frame_center_x, frame_center_y), target_center, (255, 255, 0), 2)

        cv2.putText(frame, f"Command: {command} ({yaw_command:.2f})", (20, 75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
    elif state == 'Target Lost':
        lost_frames += 1

    cv2.imshow("Trackpilot", frame)
    key = cv2.waitKey(1)

    frame_end = time.perf_counter()
    pipeline_time = frame_end - frame_start
    pipeline_times.append(pipeline_time)
    

    if key == ord('q'):
        break

if pipeline_times:
    total_frame_time = sum(pipeline_times)
    # print(f"Number of pipeline times: {len(pipeline_times)}")
    avg_frame_fps = len(pipeline_times) / total_frame_time
    print(f"Average Frame FPS: {avg_frame_fps}")

if track_times:
    total_track_time = sum(track_times)
    # print(f"Number of track times: {len(track_times)}")
    avg_track_fps = len(track_times) / total_track_time
    print(f"Average YOLO + ByteTrack FPS: {avg_track_fps}")

total_frames = tracked_frames + lost_frames
print(f"Total frames: {total_frames}")
if total_frames > 0:
    tracking_success_rate = (tracked_frames / total_frames) * 100
    print(f"Tracking success rate: {tracking_success_rate:.2f}%")

video.release()
cv2.destroyAllWindows()