from utils import (read_video, save_video, measure_distance, draw_player_stats)
from trackers import PlayerTracker, BallTracker
from courtline_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
from copy import deepcopy
import pandas as pd

def main():

    # reading the input video
    input_video_path = "input_videos/input_video.mp4"
    # breaking the video into frames
    video_frames = read_video(input_video_path)

    # tracking players in every frame by mapping their id to the coordinates of their box (in a dict) and saving all these dicts in a list for every frame
    player_tracker = PlayerTracker(model_path= "yolov8x.pt")
    player_detections = player_tracker.detect_frames(video_frames, stub_path= "tracker_stubs/player_detections.pkl")

    # tracking the ball
    ball_tracker = BallTracker(model_path= "models/yolov5nu_last.pt")
    ball_detections = ball_tracker.detect_frames(video_frames, stub_path= "tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detecting court lines
    court_line_detector = CourtLineDetector(model_path="models/keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Filtering player detections to include only the players on the court
    player_detections = player_tracker.filter_players(court_keypoints, player_detections)

    # Initialising the minicourt
    mini_court = MiniCourt(video_frames[0])

    # Detecting ball shots
    ball_shot_frames =  ball_tracker.get_ball_shot_frames(ball_detections)

    # Converting positions to minicourt positions
    player_minicourt_detections, ball_minicourt_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0
    }]

    for ball_shot_index in range(len(ball_shot_frames)-1): # not including the last frame because we would need a succeeding frame for calculations
        start_frame = ball_shot_frames[ball_shot_index]
        end_frame = ball_shot_frames[ball_shot_index + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24 # 24fps

        # get distance covered by ball
        distance_covered_by_ball_pixels = measure_distance(ball_minicourt_detections[start_frame][1], ball_minicourt_detections[end_frame][1])
        distance_covered_by_ball_meters = mini_court.pixels_to_meters(distance_covered_by_ball_pixels)

        # speed of the ball in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # player who shot the balls
        player_positions = player_minicourt_detections[start_frame]
        player_who_shot_ball = min(player_positions.keys(), key = lambda player_id: measure_distance(player_positions[player_id], ball_minicourt_detections[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_who_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_minicourt_detections[start_frame][opponent_player_id], player_minicourt_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = mini_court.pixels_to_meters(distance_covered_by_opponent_pixels)

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1]) # deep copy of previous player stats
        current_player_stats['frame_num'] = end_frame
        current_player_stats[f'player_{player_who_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_who_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_who_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left') # left joining both on frame number
    player_stats_data_df = player_stats_data_df.ffill() # replaces Nan values with the preceeding values

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots'] # because a player runs only when the opponent shoots
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']


    # Drawing bounding boxes and overlaying them on every single frame for players and the ball
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Drawing court lines on the video
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Drawing the mini court on the video
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_minicourt(output_video_frames, player_minicourt_detections)
    output_video_frames = mini_court.draw_points_on_minicourt(output_video_frames, ball_minicourt_detections, color=(0, 255, 255))

    # drawing the stats in a box
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # Drawing frame number on the to left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()