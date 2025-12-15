from trackers.tracker import Tracker
from utils.video_utils import read_video, save_video    
from utils.drawing_utils import draw_tracks
import os   

def main():
    video_frames = read_video("input_videos/input_videos.mp4")

    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(video_frames)

    # THIS WAS MISSING
    draw_tracks(video_frames, tracks)

    os.makedirs("output_videos", exist_ok=True)
    save_video(video_frames, "output_videos/output_video.avi")

main()