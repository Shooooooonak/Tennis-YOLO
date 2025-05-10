from ultralytics import YOLO
import cv2
import pickle
import os
import sys
sys.path.append('../')
from utils import measure_distance, get_centre_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_centre = get_centre_of_bbox(bbox)

            # measure min distance from each player to any of the court keypoints (might have to change criteria for choosing players)
            min_distance = float('inf')
            for i in range(0,len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_centre, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # sort the distances in ascending order
        distances.sort(key=lambda x: x[1])

        # choosing the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    
    def filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        # including only those bboxes that belong to the chosen players
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def detect_frames(self, frames, stub_path = None):
        player_detections = []

        # saving the tracked frames in a stub so that the tracker doesnt need to be run again

        if os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        else:
            print(f"Stub path not provided. Player detections will not be saved.")

        return player_detections
        
    def detect_frame(self, frame):
        results = self.model.track(frame, persist = True)[0] # persist tells model that there will be multiple frames to persist tracking
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections): # drawing bounding boxes
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) # colour in BGR format, 2 is the thickness of the rectangle
            output_video_frames.append(frame)

        return output_video_frames