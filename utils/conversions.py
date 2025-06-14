def convert_pixel_distance_to_metres(pixel_distance, reference_height_in_metres, reference_height_in_pixels):
    return pixel_distance * (reference_height_in_metres / reference_height_in_pixels)

def convert_metres_to_pixel_distance(metres, reference_height_in_metres, reference_height_in_pixels):
    return metres * (reference_height_in_pixels / reference_height_in_metres)