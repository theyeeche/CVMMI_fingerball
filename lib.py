import cv2
import mediapipe as mp
import numpy as np
import math
RADIUS = 20
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 136, 255)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


class ENVIRONMENT:
    Friction = 0.92
    STOP_Treshold = 0.02
    GamePoint = 0
    PressCounter = 0

class HandStatus:
    open = False
    center_coord=(0, 0)
    finger_coords = (0, 0)


class BallStatus:
    def __init__(self, ball_pos = (100, 100), Radius = RADIUS):
        self.ball_pos = ball_pos
        self.ball_velocity=(20, 50) 
        self.new_ball_velocity = (0, 0)
        self.ball_move = False
        self.radius = Radius
    def draw_circle(self, image, color):
        cv2.circle(image, self.ball_pos, self.radius, color, cv2.FILLED)   

    def bouncing_ball(self, image, position, velocity, deceleration):
        ball_position = np.array(position, dtype=float)
        ball_velocity = np.array(velocity, dtype=float)

        ball_position += ball_velocity

        ball_velocity *= deceleration

        for _ in range(2):  
            # 检查X方向边界
            if ball_position[0] - self.radius < 0:
                ball_position[0] = self.radius  
                ball_velocity[0] = -ball_velocity[0]
            elif ball_position[0] + self.radius > image.shape[1]:
                ball_position[0] = image.shape[1] - self.radius  
                ball_velocity[0] = -ball_velocity[0]

            # 检查Y方向边界
            if ball_position[1] - self.radius < 0:
                ball_position[1] = self.radius  
                ball_velocity[1] = -ball_velocity[1]
            elif ball_position[1] + self.radius > image.shape[0]:
                ball_position[1] = image.shape[0] - self.radius  
                ball_velocity[1] = -ball_velocity[1]

        return tuple(ball_position.astype(int)), tuple(ball_velocity.astype(int))

    


class Box:
    def __init__(self, leftup, height=100, width=100, color=WHITE, velosity = (5, 5)):
        self.leftup = leftup
        self.height = height
        self.width = width
        self.color = color
        self.velocity = velosity
        self.update_rightdown()
        self.exist = True        
        self.gameflag = False


    def update_rightdown(self):
        self.rightdown = (self.leftup[0] + self.width, self.leftup[1] + self.height)

    def draw_rectangle(self, image):
        cv2.rectangle(image, self.leftup, self.rightdown, self.color, 2, 0, 0)

    def chance_bounce_rectangle(self, image):
        height, width, _ = image.shape
        leftup = np.array(self.leftup, dtype=float)
        velocity = np.array(self.velocity, dtype=float)
        extra_boundary = self.height+300

        # 更新位置
        leftup += velocity
        # 碰撞检测并反弹
        if leftup[0] <= -extra_boundary or leftup[0] + self.width >= width+extra_boundary:
            velocity[0] = -velocity[0]
            leftup[0] = max(0, min(leftup[0], width - self.width))

        if leftup[1] <= -extra_boundary or leftup[1] + self.height >= height + extra_boundary:
            velocity[1] = -velocity[1]
            leftup[1] = max(0, min(leftup[1], height - self.height))

        self.leftup = (int(leftup[0]), int(leftup[1]))
        self.velocity = (int(velocity[0]), int(velocity[1]))
        self.update_rightdown()
        return self.leftup, self.velocity
    
    def random_box_position(self, min = 50, max = 120):
        self.leftup = np.random.randint(100, 400, size=2) 
        self.rightdown = (self.leftup[0]+np.random.randint(min, max), self.leftup[1]+np.random.randint(min, max))  


def find_index_finger_tip(image):
    '''
    input: input image
    output: finger tip coordinate
    '''
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    
    finger_tip_coords = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] # get index finger tip
            h, w, _ = image.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            finger_tip_coords = (cx, cy)
            break 
    return finger_tip_coords


def calculate_angle(a, b, c):
    # 計算三個點之間的角度
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

def detect_hand_open(image):
    '''
    input: input image
    output:
        whether the hands is opened?, hand center coordinate
    '''
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return False, None

    hand_landmarks = results.multi_hand_landmarks[0]

    # 手指尖端、指根关节和中间关节的索引
    finger_tips = [mp_hands.HandLandmark.THUMB_TIP,
                   mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]
    finger_dips = [mp_hands.HandLandmark.THUMB_IP,
                   mp_hands.HandLandmark.INDEX_FINGER_DIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                   mp_hands.HandLandmark.RING_FINGER_DIP,
                   mp_hands.HandLandmark.PINKY_DIP]
    finger_pips = [mp_hands.HandLandmark.THUMB_MCP,
                   mp_hands.HandLandmark.INDEX_FINGER_PIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                   mp_hands.HandLandmark.RING_FINGER_PIP,
                   mp_hands.HandLandmark.PINKY_PIP]

    open_fingers = 0

    for tip, dip, pip in zip(finger_tips, finger_dips, finger_pips):
        if (hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y and
            hand_landmarks.landmark[dip].y < hand_landmarks.landmark[pip].y):
            open_fingers += 1

    hand_open = open_fingers >= 4  # 判断手指伸展数量是否达到4根以上

    if hand_open:
        # 計算手掌中心
        palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        h, w, _ = image.shape
        cx, cy = int(palm_center.x * w), int(palm_center.y * h)
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # 檢查手掌的角度來確保手指是張開的
        angles = []
        for tip in finger_tips:
            tip_pos = hand_landmarks.landmark[tip]
            mcp_pos = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            wrist_pos = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            angles.append(calculate_angle(wrist_pos, mcp_pos, tip_pos))

        if all(angle > 15 for angle in angles):  # 確保所有手指與掌心的角度大於15度
            return True, (cx, cy)
        else:
            return False, None
    else:
        return False, None


def is_circle_in_rectangle(circle_center, radius, rect_top_left, rect_bottom_right):
    circle_x, circle_y = circle_center
    rect_x1, rect_y1 = rect_top_left
    rect_x2, rect_y2 = rect_bottom_right

    # 檢查圓心是否在矩形內
    if rect_x1 <= circle_x <= rect_x2 and rect_y1 <= circle_y <= rect_y2:
        return True

    # 計算圓心到矩形四條邊的最短距離
    closest_x = np.clip(circle_x, rect_x1, rect_x2)
    closest_y = np.clip(circle_y, rect_y1, rect_y2)
    distance_x = circle_x - closest_x
    distance_y = circle_y - closest_y

    distance_squared = distance_x ** 2 + distance_y ** 2

    return distance_squared <= radius ** 2

def line_equation_from_points(x1, y1, x2, y2):
    try:        
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b
    except:
        return float('inf'), float('inf')

def reflect(direction, normal):
    """反射向量"""
    return direction - 2 * np.dot(direction, normal) * normal

def draw_reflected_line(image, start_point, direction):
    height, width, _ = image.shape
    direction = direction / np.linalg.norm(direction) 
    # 计算射线的终点
    end_point = start_point + direction * 512 

    # 检测与边界的碰撞并计算反射点
    m, b =line_equation_from_points(start_point[0], start_point[1], end_point[0], end_point[1])
    if end_point[0] <= 0:
        end_point[0] = 0
        end_point[1] = b
    elif end_point[0] >= width:
        end_point[0] = width
        end_point[1] = width*m+b
    if end_point[1] <= 0:
        end_point[0] = -b/m
        end_point[1] = 0
    elif end_point[1] >= height:
        end_point[0] = (height - b) / m
        end_point[1] = height

    # 计算法向量并反射
    normal = np.array([0, 0])
    if end_point[0] == 0:
        normal = np.array([1, 0])
    elif end_point[0] == width:
        normal = np.array([-1, 0])
    if end_point[1] == 0:
        normal = np.array([0, 1])
    elif end_point[1] == height:
        normal = np.array([0, -1])

    if np.linalg.norm(normal) != 0:  
        new_direction = reflect(direction, normal)
        reflected_end_point = end_point + new_direction * 512
        cv2.line(image, tuple(start_point.astype(int)), tuple(end_point.astype(int)), (0, 255, 0), 2)
        cv2.line(image, tuple(end_point.astype(int)), tuple(reflected_end_point.astype(int)), (0, 0, 255), 2)
    else:
        cv2.line(image, tuple(start_point.astype(int)), tuple(end_point.astype(int)), (0, 255, 0), 2)

    return image

def draw_line_to_circle(image, circle_center, finger_point):
    circle_center = np.array(circle_center)
    finger_point = np.array(finger_point)
    try:
        direction =  circle_center - finger_point
    except:
        direction = (0, 0)
    image = draw_reflected_line(image, circle_center, direction)

    return image