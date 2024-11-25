from utils import (read_video,
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meter)

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from copy import deepcopy
import constants
import cv2
import pandas as pd

def main():
    #Read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    #Detect player and ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolov5_best.pt')
    
    player_detection = player_tracker.detect_frames(video_frames, 
                                                    read_from_stub=True, 
                                                    stub_path="tracker_stubs/player_detection.pkl")
    
    ball_detection = ball_tracker.detect_frames(video_frames, 
                                                    read_from_stub=True, 
                                                    stub_path="tracker_stubs/ball_detection.pkl")
    ball_detection = ball_tracker.interpolate_ball_positions(ball_detection)
    #court line detector model
    court_model_path = 'models/keypoint_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #choose players
    player_detection = player_tracker.choose_and_filter_players(court_keypoints, player_detection)

    # miniCourt
    mini_court = MiniCourt(video_frames[0])

    # Detecte ball shot
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detection)

    #Convert position to mini court position
    player_mini_court_detection, ball_mini_court_detection = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detection, ball_detection, court_keypoints)

    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    }]

    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24
        #get distance covered by th eball
        distance_covered_by_ball_in_pixels = measure_distance(ball_mini_court_detection[start_frame][1], ball_mini_court_detection[end_frame][1])
        distance_covered_by_ball_in_meters = convert_pixel_distance_to_meter(distance_covered_by_ball_in_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court())

        #speed of the ball in km/h

        speed_of_ball_shot = distance_covered_by_ball_in_meters/ball_shot_time_in_seconds * 3.6

        #player xho shot the ball
        player_positions = player_mini_court_detection[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id], ball_detection[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detection[start_frame][opponent_player_id],
                                                               player_mini_court_detection[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meter(distance_covered_by_opponent_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court())

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds *3.6

        current_player_stat = deepcopy(player_stats_data[-1])
        current_player_stat['frame_num'] = start_frame
        current_player_stat[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stat[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stat[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stat[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stat[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stat)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']

    

    #Draw output
    ##Draw bounding boxes
    output_video_frame = player_tracker.draw_bboxes(video_frames, player_detection)
    output_video_frame = ball_tracker.draw_bboxes(output_video_frame, ball_detection)
    
    ## Draw court keypoint
    output_video_frame = court_line_detector.draw_keypoints_on_video(output_video_frame, court_keypoints)
    ## Draw minicout
    output_video_frame = mini_court.draw_mini_court(output_video_frame)
    ## Draw player and ball point minicourt
    output_video_frame = mini_court.draw_points_on_mini_court(output_video_frame, player_mini_court_detection, color=(0, 0, 0))
    output_video_frame = mini_court.draw_points_on_mini_court(output_video_frame, ball_mini_court_detection, color =(0, 255, 255))

    ##Draw player stats
    output_video_frame = draw_player_stats(output_video_frame, player_stats_data_df)

    ## Draw frame number on the top left corner
    for i , frame in enumerate(output_video_frame):
        cv2.putText(frame, f"Frame : {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frame, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()