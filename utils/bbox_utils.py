def get_centre_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    centre_x = int((x1 + x2) / 2)
    centre_y = int((y1 + y2) / 2)
    return (centre_x, centre_y)

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_foot_positions(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2 ), y2 )

def get_closest_keypoint_index(foot_position, keypoints, keypoint_indices):
    closest_distance = float('inf')
    closest_keypoint_index = keypoint_indices[0]

    for keypoint_index in keypoint_indices:
        keypoint_position = (keypoints[keypoint_index * 2], keypoints[keypoint_index * 2 + 1])
        distance = abs(foot_position[1] - keypoint_position[1]) 

        if distance < closest_distance:
            closest_distance = distance
            closest_keypoint_index = keypoint_index

    return closest_keypoint_index

def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]

def measure_xy_distance(p1, p2):
    distance_x = abs(p1[0] - p2[0])
    distance_y = abs(p1[1] - p2[1])
    return distance_x, distance_y

def get_centre_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    centre_x = int((x1 + x2) / 2)
    centre_y = int((y1 + y2) / 2)
    return (centre_x, centre_y)