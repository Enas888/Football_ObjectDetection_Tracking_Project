# football analysis using YOLO model

📌 Project Pipeline Overview

This project implements an end-to-end football video analysis pipeline that performs object detection, multi-object tracking, and visualization using a deep learning model.

🔹 1. Video Input

The pipeline starts by loading a football match video.

The video is read frame-by-frame into memory for processing.

🔹 2. Object Detection (YOLO)

A pretrained YOLO model is used to detect objects in each frame.

Target classes include:

- Players

- Referees

- Ball

Frames are processed in batches to reduce memory usage and improve performance.

## Example of one frame:

<img width="1540" height="808" alt="image" src="https://github.com/user-attachments/assets/6ee2d22e-f371-4849-a214-1e557c0ff3c9" />


🔹 3. Object Tracking (ByteTrack)

Detected objects are passed to ByteTrack for multi-object tracking.

Each object is assigned a unique tracking ID that remains consistent across frames.

Goalkeepers are reclassified as players to simplify player analysis.

🔹 4. Track Management

Tracking results are stored in a structured format:

- Separate tracks for players, referees, and the ball

- Bounding boxes indexed by frame number and track ID

🔹 5. Visualization

Bounding boxes and tracking IDs are drawn on video frames:

- Players → Green

- Referees → Red

- Ball → Blue

Drawing is done directly on the frames to prepare them for export.

🔹 6. Video Output

- The processed frames are saved as a new video file.

- The output video preserves the original frame rate and resolution while adding visual tracking annotations.

✅ Final Output

The final result is an annotated football video showing tracked players, referees, and the ball with consistent IDs across frames, ready for further analysis or visualization.

<img width="1554" height="816" alt="image" src="https://github.com/user-attachments/assets/2e898384-e3fd-443a-a7f0-fb3cb55d7b87" />


