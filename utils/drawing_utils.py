import cv2

def draw_tracks(frames, tracks):
    for frame_idx, frame in enumerate(frames):

        # ---- Players ----
        for track_id, player in tracks["players"][frame_idx].items():
            x1, y1, x2, y2 = map(int, player["bbox"])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"P {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ---- Referees ----
        for track_id, ref in tracks["referees"][frame_idx].items():
            x1, y1, x2, y2 = map(int, ref["bbox"])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, f"R {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # ---- Ball ----
        for _, ball in tracks["ball"][frame_idx].items():
            x1, y1, x2, y2 = map(int, ball["bbox"])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, "Ball", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    