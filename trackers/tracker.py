from ultralytics import YOLO
import supervision as sv
import os
import pickle
import cv2
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 

        # detections = self.model.predict(frames, conf=0.1)
        for i in range(0,len(frames),batch_size):  #batch size to handle memory issues
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[], 
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names                         # {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            # detection_supervision: [Bounding Box, Mask, Confidence, Class ID, Track ID]
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            # detection_supervision [class_id]: List of class IDs for each detection

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # detection_with_tracks: [Bounding Box, Mask, Confidence, Class ID, Track ID]

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)  
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)), # (the major axis, the minor axis)
            angle=0.0,
            startAngle=-45,  # the cicle is not completed
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        return frame

    def draw_annotations(self,video_frames, tracks, team_ball_control):
            output_video_frames= []
            for frame_num, frame in enumerate(video_frames):
                frame = frame.copy()

                player_dict = tracks["players"][frame_num]
                ball_dict = tracks["ball"][frame_num]
                referee_dict = tracks["referees"][frame_num]

                # Draw Players
                for track_id, player in player_dict.items():
                    color = player.get("team_color",(0,0,255))
                    frame = self.draw_ellipse(frame, player["bbox"],color, track_id)
                    if player.get('has_ball',False):
                        frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

                # Draw Referees
                for track_id, referee in referee_dict.items():
                    #yellow
                    color = (0,255,255)
                    frame = self.draw_ellipse(frame, referee["bbox"],color)
                    
                # Draw Ball 
                for track_id, ball in ball_dict.items():
                    color = (0,255,0)
                    frame = self.draw_traingle(frame, ball["bbox"], color)

                # Draw Team Ball Control
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
                output_video_frames.append(frame)

            return output_video_frames
    
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    
    def interpolate_ball_positions(self,ball_positions):
        # 1 is the track id so if the ball is exist in this frame the value will be 1 and if missing the value will be {} 
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        
        # create a data frame of 4 columns of all the postions of the ball          
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()   # fill the missing
     
        # If the first frame is missing annotate it with the nearest frame we find with the backfill
        df_ball_positions = df_ball_positions.bfill()


        # put the positions with the interplated values into a dict of {id=1 : bbox=['x1','y1','x2','y2']
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )   # (frame, position of the top left, position of the bottom right, color, -1 is for filled or write cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # (The mask, alpha of mask, the original image, the alpha of the orginal, gamma for brightness adjustment, destination image acts like inplace)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)  # (frame, text, pos, font, font scale, color, thickness) 
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)  
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)), # (the major axis, the minor axis)
            angle=0.0,
            startAngle=-45,  # the cicle is not completed
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame