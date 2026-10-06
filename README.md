# Trackpilot

Computer vision system for detecting and tracking a selected person in video using YOLO11n and ByteTrack.

## Demo
https://github.com/user-attachments/assets/db51d7f1-ce83-46d7-b977-6ec84034f34c

## Motivation

I built trackpilot to learn how object detection and tracking work together in a computer vision system.

Object detection can identify a person in each frame, but tracking is needed to maintain that person's identity across frames. trackpilot uses YOLO11n for person detection and ByteTrack to maintain tracking IDs.

The tracking information is then used to estimate the target's position and distance from the camera and generate movement commands.

## Pipeline

For each video frame, trackpilot:

1. Reads the frame using OpenCV
2. Runs YOLO11n with ByteTrack
3. Finds the selected target using its tracking ID
4. Calculates the target's center and relative size
5. Determines the target's position relative to the frame
6. Generates yaw and distance commands

The current system uses a recorded video as its input.

## Tracking

trackpilot only considers people as targets.

When tracking begins, the first tracked person is assigned as the target. The system then uses that person's ByteTrack ID to distinguish the target from other people in later frames.

For the selected target, the system calculates:

- Bounding box coordinates
- Center position
- Bounding box area relative to the frame
- Horizontal position error
- Distance error

A deadband is used for both horizontal position and distance so that small changes do not constantly cause new commands.

## Control

The target's position relative to the center of the frame is used to generate yaw commands.

The target's size relative to the frame is used as a rough measure of distance.

The system can generate commands including:

```text
Yaw Left
Yaw Right
Hold
Move forward
Move backward
Hold distance
Stop
```

The yaw and distance commands are limited to a range of `-1.0` to `1.0`.

## Results

trackpilot was tested using a 408-frame recorded video.

| Metric | Result |
| --- | ---: |
| Source video FPS | 28.90 FPS |
| End-to-end processing | 20.30 FPS |
| YOLO + ByteTrack | 25.94 FPS |
| Tracking success rate | 96.08% |
| Unique target IDs | 1 |

The YOLO + ByteTrack portion of the pipeline ran at 25.94 FPS, while the full pipeline including video processing and display ran at 20.30 FPS.

The target was successfully tracked in 96.08% of the frames where tracking state was recorded.

## Project Structure

```text
trackpilot/
├── scripts/
│   └── read_video.py
├── src/
│   └── trackpilot/
│       ├── __init__.py
│       ├── metrics.py
│       └── pipeline.py
├── videos/
│   ├── input/
│   │   ├── .gitkeep
│   │   └── people.mov
│   └── output/
│       └── .gitkeep
├── .gitignore
├── requirements.txt
├── yolo11n.pt
└── README.md
```

## Running

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python scripts/read_video.py
```

The input video is:

```text
videos/input/people.mov
```

Press `q` to stop the video.

After the video finishes, the program prints the tracking metrics to the terminal.

## Technologies

- Python
- OpenCV
- Ultralytics YOLO11n
- ByteTrack

## Why I Built It

I wanted to go beyond running an object detection model and understand how detection, tracking, and control logic fit together.

This project gave me experience working with bounding boxes, tracking IDs, frame processing, tracking metrics, and turning visual information into movement commands.