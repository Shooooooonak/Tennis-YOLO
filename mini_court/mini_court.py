import cv2
import sys

sys.path.append('../')
import constants
from utils import (
    convert_pixel_distance_to_metres, 
    convert_metres_to_pixel_distance, 
    get_foot_positions, 
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_centre_of_bbox,
    measure_distance)
import numpy as np

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def metres_to_pixels(self, metres):
        return convert_metres_to_pixel_distance(metres, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)
    
    def pixels_to_meters(self, pixels):
        return convert_pixel_distance_to_metres(pixels, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)

    def set_court_drawing_keypoints(self):
        drawing_keypoints = [0]*28

        # point 0
        drawing_keypoints[0], drawing_keypoints[1] = int(self.court_start_x), int(self.court_start_y)

        # point 1
        drawing_keypoints[2], drawing_keypoints[3] = int(self.court_end_x), int(self.court_start_y)

        # point 2
        drawing_keypoints[4] = int(self.court_start_x)
        drawing_keypoints[5] = self.court_start_y + self.metres_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)

        # point 3
        drawing_keypoints[6] = drawing_keypoints[0] + self.court_drawing_width
        drawing_keypoints[7] = drawing_keypoints[5]

        # point 4
        drawing_keypoints[8] = drawing_keypoints[0] + self.metres_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_keypoints[9] = drawing_keypoints[1]

        # point 5
        drawing_keypoints[10] = drawing_keypoints[4] + self.metres_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_keypoints[11] = drawing_keypoints[5]

        # point 6
        drawing_keypoints[12] = drawing_keypoints[2] - self.metres_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_keypoints[13] = drawing_keypoints[3]

        # point 7
        drawing_keypoints[14] = drawing_keypoints[6] - self.metres_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_keypoints[15] = drawing_keypoints[7]

        # point 8
        drawing_keypoints[16] = drawing_keypoints[8]
        drawing_keypoints[17] = drawing_keypoints[9] + self.metres_to_pixels(constants.NO_MANS_LAND_HEIGHT)

        # point 9
        drawing_keypoints[18] = drawing_keypoints[16] + self.metres_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[19] = drawing_keypoints[17]

        # point 10
        drawing_keypoints[20] = drawing_keypoints[10]
        drawing_keypoints[21] = drawing_keypoints[11] - self.metres_to_pixels(constants.NO_MANS_LAND_HEIGHT)

        # point 11
        drawing_keypoints[22] = drawing_keypoints[20] + self.metres_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[23] = drawing_keypoints[21]

        # point 12
        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18]) / 2)
        drawing_keypoints[25] = drawing_keypoints[17]

        # point 13
        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22]) / 2)
        drawing_keypoints[27] = drawing_keypoints[21]

        self.drawing_keypoints = drawing_keypoints
        
    def set_court_lines(self):
        self.lines = [
            (0,2),
            (4,5),
            (6,7),
            (1,3),

            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (10,11),
            (2,3)
        ]
   
    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.drawing_rectangle_height + self.buffer

        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court_points_on_minicourt(self, frame):

        # drawing the points

        for i in range(0, len(self.drawing_keypoints), 2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # drawing the lines

        for line in self.lines:
            start_point = (int(self.drawing_keypoints[line[0]*2]), int(self.drawing_keypoints[line[0]*2 + 1]))
            end_point = (int(self.drawing_keypoints[line[1]*2]), int(self.drawing_keypoints[line[1]*2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # drawing the net

        net_start_point = (self.drawing_keypoints[0], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        net_end_point = (self.drawing_keypoints[2], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 255, 255), 2)

        return frame
    
    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)

        # Drawing the background rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
        
        output_frame = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        output_frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return output_frame
    
    def draw_mini_court(self, frames):
        output_frames = []
        
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court_points_on_minicourt(frame)
            output_frames.append(frame)

        return output_frames
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    def get_court_drawing_keypoints(self):
        return self.drawing_keypoints
    
    def get_mini_court_coordinates(self, 
                                   object_position, 
                                   closest_keypoint, 
                                   closest_keypoint_index, 
                                   player_height_in_pixels, 
                                   player_height_in_metres
                                   ):
        
        # Get the distance between closest keypoint and player

        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_keypoint)

        # Converting pixel distance to metres
        distance_from_keypoint_x_metres = convert_pixel_distance_to_metres(distance_from_keypoint_x_pixels, player_height_in_metres, player_height_in_pixels)
        distance_from_keypoint_y_metres = convert_pixel_distance_to_metres(distance_from_keypoint_y_pixels, player_height_in_metres, player_height_in_pixels)
        
        # Convert to minicourt coordinates
        mini_court_x_distance_pixels = self.metres_to_pixels(distance_from_keypoint_x_metres)
        mini_court_y_distance_pixels = self.metres_to_pixels(distance_from_keypoint_y_metres)

        closest_minicourt_keypoint = (self.drawing_keypoints[closest_keypoint_index*2], self.drawing_keypoints[closest_keypoint_index*2 + 1])
        
        minicourt_player_position = (closest_minicourt_keypoint[0] + mini_court_x_distance_pixels, closest_minicourt_keypoint[1] + mini_court_y_distance_pixels)
        
        return minicourt_player_position
    
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_keypoints):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS,
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            output_player_bboxes_dict = {}

            ball_box = ball_boxes[frame_num][1]
            ball_position = get_centre_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda player_id: measure_distance(ball_position, get_centre_of_bbox(player_bbox[player_id])))
            
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_positions(bbox)

                # Get the closest keypoint in pixels
                closest_keypoint_index = get_closest_keypoint_index(foot_position, original_court_keypoints, [0,2,12,13]) # we chose the reference keypoints, left end and centre line of both sides of court
                closest_keypoint = (original_court_keypoints[closest_keypoint_index*2], 
                                    original_court_keypoints[closest_keypoint_index*2 + 1])
                
                # Get the player height in pixels
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                minicourt_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_keypoint,  
                                                                            closest_keypoint_index,
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                
                output_player_bboxes_dict[player_id] = minicourt_player_position

                if closest_player_id_to_ball == player_id:
                    closest_keypoint_index = get_closest_keypoint_index(ball_position, original_court_keypoints, [0,2,12,13]) 
                    closest_keypoint = (original_court_keypoints[closest_keypoint_index*2], 
                                        original_court_keypoints[closest_keypoint_index*2 + 1])
                    
                    minicourt_player_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_keypoint,  
                                                                            closest_keypoint_index,
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                    output_ball_boxes.append({1:minicourt_player_position})
            output_player_boxes.append(output_player_bboxes_dict)
            
        return output_player_boxes, output_ball_boxes
    
    def draw_points_on_minicourt(self, frames, positions, color = (0, 255, 0)):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for _ , position in positions[frame_num].items():
                x, y = int(position[0]), int(position[1])
                cv2.circle(frame, (x, y), 5, color, -1)
            output_frames.append(frame)
        return output_frames