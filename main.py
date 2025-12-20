from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
import os 
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
import numpy as np

def main():
    video_frames = read_video('input_videos/input_videos.mp4')

    trackers = Tracker('models/best.pt')
    os.makedirs('stubs', exist_ok=True)

    tracks = trackers.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    # Interpolate Ball Positions
    tracks["ball"] = trackers.interpolate_ball_positions(tracks["ball"])

    team_assigner = TeamAssigner()

    # ✅ FIX: train team colors using ONE frame
    team_assigner.assign_team_color(
        video_frames[0],
        tracks['players'][0]
    )

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()

    team_ball_control= []

    # Loop over all the frames to to get the assigned player in each frame
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])

        # No player have the ball so we will assign to the last team who has the ball    
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control= np.array(team_ball_control)
    

    video_frames = trackers.draw_annotations(video_frames, tracks,team_ball_control)


    os.makedirs('output_videos', exist_ok=True)
    save_video(video_frames, 'output_videos/output_video.avi')

main()