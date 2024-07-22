import cv2
import sys
import numpy as np
import collections
sys.path.append("../")
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_to_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance,
)

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20
        self.ball_positions = collections.deque(maxlen=5)

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()  

    def convert_meters_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width)

    def set_court_drawing_keypoints(self):
        drawing_keypoints = [0]*28

        # Point 0 - Top left corner
        drawing_keypoints[0], drawing_keypoints[1] = (int(self.court_start_x), int(self.court_start_y))
        # Point 1 - Top right corner
        drawing_keypoints[2], drawing_keypoints[3] = (int(self.court_end_x), int(self.court_start_y))
        # Point 2 - Down left corner
        drawing_keypoints[4] = (int(self.court_start_x))
        drawing_keypoints[5] = self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # Point 3 - Down right corner
        drawing_keypoints[6] = drawing_keypoints[0] + self.court_drawing_width
        drawing_keypoints[7] = drawing_keypoints[5]
        # Point 4
        drawing_keypoints[8] = drawing_keypoints[0] +  self.convert_meters_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[9] = drawing_keypoints[1] 
        # Point 5
        drawing_keypoints[10] = drawing_keypoints[4] + self.convert_meters_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[11] = drawing_keypoints[5] 
        # Point 6
        drawing_keypoints[12] = drawing_keypoints[2] - self.convert_meters_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[13] = drawing_keypoints[3] 
        # Point 7
        drawing_keypoints[14] = drawing_keypoints[6] - self.convert_meters_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[15] = drawing_keypoints[7] 
        # Point 8
        drawing_keypoints[16] = drawing_keypoints[8] 
        drawing_keypoints[17] = drawing_keypoints[9] + self.convert_meters_pixels(constants.NO_MANS_LAND_HEIGHT)
        # Point 9
        drawing_keypoints[18] = drawing_keypoints[16] + self.convert_meters_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[19] = drawing_keypoints[17] 
        # Point 10
        drawing_keypoints[20] = drawing_keypoints[10] 
        drawing_keypoints[21] = drawing_keypoints[11] - self.convert_meters_pixels(constants.NO_MANS_LAND_HEIGHT)
        # Point 11
        drawing_keypoints[22] = drawing_keypoints[20] +  self.convert_meters_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[23] = drawing_keypoints[21] 
        # Point 12
        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18])/2)
        drawing_keypoints[25] = drawing_keypoints[17] 
        # Point 13
        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22])/2)
        drawing_keypoints[27] = drawing_keypoints[21] 

        self.drawing_keypoints = drawing_keypoints

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self,frame):
        for i in range(0, len(self.drawing_keypoints),2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_keypoints[line[0]*2]), int(self.drawing_keypoints[line[0]*2+1]))
            end_point = (int(self.drawing_keypoints[line[1]*2]), int(self.drawing_keypoints[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_keypoints[0], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        net_end_point = (self.drawing_keypoints[2], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.6
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, alpha, 0)[mask]

        return out

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Convert pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_to_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_to_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = (self.drawing_keypoints[closest_key_point_index*2],
                                        self.drawing_keypoints[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        first_frame_detections = next(iter(player_boxes), {})
        player_ids = list(first_frame_detections.keys())

        if len(player_ids) < 2:
            raise ValueError("Expected detections for at least 2 players")

        # Create the player_heights dictionary dynamically
        player_heights = {
            player_ids[0]: constants.PLAYER_1_HEIGHT_METERS,
            player_ids[1]: constants.PLAYER_2_HEIGHT_METERS
        }


        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get the closest keypoint in pixels (0 top left, 2 bottom left, 12 middle top, 13 middle bottom)
                closest_key_point_index = get_closest_keypoint_index(foot_position, original_court_key_points, [0, 2, 12, 13])
                closest_key_point = (original_court_key_points[closest_key_point_index * 2],
                                    original_court_key_points[closest_key_point_index * 2 + 1])

                # Get Player height in pixels (maximum height in the last 50 frames)
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i].get(player_id, bbox)) for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point,
                                                                            closest_key_point_index,
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )

                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get the closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(ball_position, original_court_key_points, [0, 2, 12, 13])
                    closest_key_point = (original_court_key_points[closest_key_point_index * 2],
                                        original_court_key_points[closest_key_point_index * 2 + 1])

                    mini_court_player_position = self.get_mini_court_coordinates(ball_position,
                                                                                closest_key_point,
                                                                                closest_key_point_index,
                                                                                max_player_height_in_pixels,
                                                                                player_heights[player_id]
                                                                                )
                    output_ball_boxes.append({1: mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes

    def update_ball_position(self, ball_position):
        # Add the new position to the deque
        self.ball_positions.append(ball_position)

        # Calculate the average position
        avg_x = sum(pos[0] for pos in self.ball_positions) / len(self.ball_positions)
        avg_y = sum(pos[1] for pos in self.ball_positions) / len(self.ball_positions)

        return (avg_x, avg_y)



    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                if color == (0, 255, 255):  # If drawing the ball (yellow color)
                    position = self.update_ball_position(position)
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, color, -1)
        return frames


    