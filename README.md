# ⚽ Football Object Detection & Tracking with Advanced Analytics

This project implements an end-to-end football video analytics pipeline using YOLOv8, enabling player and ball detection, multi-object tracking, team assignment, and ball possession analysis, with annotated video output.

# 🚀 Features

🧠 YOLOv8-based object detection (players, ball)

🎯 Multi-object tracking across video frames

🎨 Automatic team assignment based on jersey color

⚽ Ball-to-player assignment (ball possession detection)

📊 Team ball control analysis over time

🎥 Annotated output video with bounding boxes, team colors, and possession indicators

# 📁 Project Structure

```text
Football_ObjectDetection_Tracking_Project/
│
├── main.py
├── models/
│   └── best.pt                  # Trained YOLOv8 model
│
├── input_videos/
│   └── input_videos.mp4         # Input football match video
│
├── output_videos/
│   └── output_video.avi         # Final annotated output
│
├── stubs/
│   └── track_stubs.pkl          # Cached tracking results
│
├── trackers/
│   └── tracker.py               # Detection & tracking logic
│
├── team_assigner/
│   └── team_assigner.py         # Team color detection & assignment
│
├── player_ball_assigner/
│   └── player_ball_assigner.py  # Ball possession logic
│
├── utils/
│   └── video_utils.py           # Video read/write utilities
│
└── README.md
```


# 🧠 Pipeline Overview

The system processes a football video through the following steps:

Video Loading

Reads input video and extracts frames.

Object Detection & Tracking

Uses YOLOv8 to detect players and the ball.

Tracks objects across frames.

Supports cached tracking using stub files to speed up experiments.

Ball Trajectory Interpolation

Smooths missing ball detections for consistent tracking.

Team Assignment

Learns team colors from the first frame.

Assigns each player to a team dynamically.

Ball Possession Detection

Assigns the ball to the closest player per frame.

Determines which team controls the ball over time.

Visualization & Annotation

Draws bounding boxes, team colors, and possession indicators.

Saves an annotated output video.

# ▶️ How to Run
## 1️⃣ Install Dependencies
pip install ultralytics opencv-python numpy


Make sure you have Python 3.8+

## 2️⃣ Prepare Files

Place your input video in:

input_videos/input_videos.mp4


Place your trained YOLOv8 model in:

models/best.pt

## 3️⃣ Run the Pipeline
python main.py

## 4️⃣ Output

Annotated video will be saved to:

output_videos/output_video.avi

# 📊 Output Visualizations

The output video includes:

Player and ball bounding boxes

Team-specific colors

Ball possession indicators

Team-level ball control timeline

# 🧪 Technologies Used

Python

YOLOv8 (Ultralytics)

OpenCV

NumPy

Multi-Object Tracking

Computer Vision for Sports Analytics

#🔮 Future Improvements

Player re-identification across camera cuts

Tactical heatmaps and formation analysis

Real-time inference optimization

Event detection (passes, shots, goals)
