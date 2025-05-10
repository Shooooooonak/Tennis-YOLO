from ultralytics import YOLO
import cv2
import pickle
import os
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_detections):
        ball_positions = [x.get(1,[]) for x in ball_detections]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # interpolating missing ball positions for empty bounding boxes in frames
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() # fills missing values with the next valid observation

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()] # 1 is the ball track id

        return ball_positions
    
    def get_ball_shot_frames(self, ball_detections):

        # converting ball detections to a df
        ball_positions = [x.get(1,[]) for x in ball_detections]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # pasting code from ball_analysis.ipynb
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window = 5, min_periods = 1, center = False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        df_ball_positions['ball_hit'] = 0

        minimum_change_frames_for_hit = 25 # minimum number of frames the ball changes position for after an impulse
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit*1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i+1, i + int(minimum_change_frames_for_hit*1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1


                if change_count >  minimum_change_frames_for_hit -1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        return frame_nums_with_ball_hits


    def detect_frames(self, frames, stub_path = None):
        ball_detections = []

        # saving the tracked frames in a stub so that the tracker doesnt need to be run again

        if os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        else:
            print(f"Stub path not provided. Ball detections will not be saved.")

        return ball_detections
        
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf = 0.2)[0] # not tracking because there's only one ball
      
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            
            ball_dict[1] = result

        return ball_dict
    
    def draw_bboxes(self, video_frames, ball_detections): # drawing bounding boxes
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2) # colour in BGR format, 2 is the thickness of the rectangle
            output_video_frames.append(frame)

        return output_video_frames