import cv2
import time
from ultralytics import YOLO

class Pipeline:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self.target_id = None

        self.confidence_threshold = 0.25

        self.yaw_deadband = 0.05
        self.yaw_kp = 0.8

        self.desired_area_ratio = 0.12
        self.distance_deadband = 0.01
        self.distance_kp = 3.0

    def process_frame(self, frame, metrics):
        target_found = False
        target_center = None
        target_area_ratio = None

        yaw_command = 0.0
        command = "Stop"

        forward_command = 0.0
        distance_command = "Stop"

        track_time_start = time.perf_counter()
        
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=self.confidence_threshold, verbose=False)
    
        track_time_end = time.perf_counter()
        track_time = track_time_end - track_time_start
    
        metrics.record_track_time(track_time)

        result = results[0]

        for box in result.boxes:
            if box.id is None:
                continue
    
            track_id = int(box.id[0])
    
            if self.target_id is None:
                self.target_id = track_id
    
            is_target = track_id == self.target_id
    
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
    
                metrics.record_target_id(self.target_id)
            else:
                color = (0, 255, 0)
                label = f"ID {track_id} | {confidence:.2f}"
    
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
            cv2.putText(frame, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
    
        if self.target_id is None:
            state = 'Searching'
        elif target_found:
            state = 'Tracking'
        else:
            state = 'Target Lost'
    
        state_color = (0, 255, 0) if state == 'Tracking' else (0, 0, 255)
    
        cv2.putText(frame, f"State: {state}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, state_color, 2)

        metrics.record_tracking_state(state)
        
        if state == 'Tracking':
            cv2.putText(frame, f"Target size: {target_area_ratio:.3f}", (20, 110), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
    
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2
    
            error_x = target_center_x - frame_center_x
            normalized_error_x = error_x / (frame_width / 2)
    
            if abs(normalized_error_x) <= self.yaw_deadband:
                yaw_command = 0.0
                command = 'Hold'
            else:
                yaw_command = self.yaw_kp * normalized_error_x
                yaw_command = max(-1.0, min(1.0, yaw_command))
    
                if yaw_command < 0:
                    command = 'Yaw Left'
                else:
                    command = 'Yaw Right'
    
            distance_error = self.desired_area_ratio - target_area_ratio
    
            if abs(distance_error) <= self.distance_deadband:
                forward_command = 0.0
                distance_command = 'Hold distance'
            else:
                forward_command = self.distance_kp * distance_error
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

        return frame
