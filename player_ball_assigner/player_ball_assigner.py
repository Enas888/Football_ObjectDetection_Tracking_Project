from utils.bbox_utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70    # the distance that by max acceptable to assign the ball
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        # loop over all the players to assign the ball to one of them
        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[3]),ball_position)   # left foot
            distance_right = measure_distance((player_bbox[2],player_bbox[3]),ball_position)  # right foot
            
            # The nearest foot
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:   # the distance is less than the th
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player	# the id of the nearest  player to the ball