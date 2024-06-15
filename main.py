import cv2
from lib import *

def main():
    cap = cv2.VideoCapture(0)
    bs = BallStatus(Radius=20)
    bx1 = Box((300, 300), color= GREEN)
    redbx = Box((100, 200), color=RED)
    chance_bx = Box(leftup=(500, 50), color=ORANGE, height=25, width=25, velosity= (10, 0))
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    r, g, b =255, 0, 127
    while True:
        success, image = cap.read()
        if not success:
            print("pass none")
            break
        image = cv2.flip(image, 1)


        # ball is moving
        if bs.ball_move: 
            bs.ball_pos, bs.new_ball_velocity= bs.bouncing_ball(image,bs.ball_pos, bs.ball_velocity, ENVIRONMENT.Friction)
        else: #ball is not moving
            HandStatus.finger_coords = find_index_finger_tip(image)
            HandStatus.open, HandStatus.center_coord = detect_hand_open(image)        

        if ENVIRONMENT.PressCounter %10 == 0 and redbx.gameflag == True:
            redbx.gameflag = False


        # check the ball meet the box event
        if is_circle_in_rectangle(bs.ball_pos, bs.radius, bx1.leftup, bx1.rightdown) and HandStatus.open == False:
            ENVIRONMENT.GamePoint+=10           
            bx1.random_box_position(50, 150)
        if is_circle_in_rectangle(bs.ball_pos, bs.radius, redbx.leftup, redbx.rightdown) and HandStatus.open == False and redbx.gameflag == False:
            ENVIRONMENT.GamePoint-=10            
            redbx.random_box_position(50, 120)
        if is_circle_in_rectangle(bs.ball_pos, bs.radius, chance_bx.leftup, chance_bx.rightdown) and HandStatus.open == False:
            chance_bx.velocity = np.random.randint(-10, 10, size=2)
            redbx.gameflag = True

        # if hand is open, set ball pos to hand center
        if HandStatus.open: 
            bs.ball_pos = HandStatus.center_coord
            bs.ball_move = False
            redbx.random_box_position(50, 120)
            bx1.random_box_position(50, 150)
        # else put rectangle on the screen
        else: 
            if redbx.gameflag == False:
                redbx.draw_rectangle(image)  
                chance_bx.chance_bounce_rectangle(image)
                chance_bx.draw_rectangle(image) 
            bx1.draw_rectangle(image)   
        
        # Set ball to stop mode when the velosity is slower than threshold
        if (abs(bs.ball_velocity[0])-abs(bs.new_ball_velocity[0]) < ENVIRONMENT.STOP_Treshold) or (abs(bs.ball_velocity[1])-abs(bs.new_ball_velocity[1]) < ENVIRONMENT.STOP_Treshold):
            bs.ball_move=False
            if redbx.gameflag == False:
                bs.draw_circle(image, color=GREEN)
            else:
                r = (r+10)%255
                g = (g+10)%255
                b = (b+10)%255
                bs.draw_circle(image, color=(b, g, r))
                # image = draw_line_to_circle(image, bs.ball_pos, HandStatus.finger_coords)
        else:
            bs.ball_velocity = bs.new_ball_velocity
            bs.draw_circle(image, color=RED)

        # DRAW line when hand is close and the ball is not moving
        if bs.ball_move == False and HandStatus.open == False and HandStatus.finger_coords != None:
            cv2.line(image, bs.ball_pos, HandStatus.finger_coords, BLACK, 2, 0)
            image = draw_line_to_circle(image, bs.ball_pos, HandStatus.finger_coords)
        cv2.putText(image, str(ENVIRONMENT.GamePoint), (20, 50), 1, 3, RED, 2, 1)
        cv2.imshow('show', image)
    
        # next play setting
        key = cv2.waitKey(5)
        if key == ord(' ') and bs.ball_move ==False:
            bs.ball_move = True
            ENVIRONMENT.PressCounter +=1
            try:
                f_x = int(HandStatus.finger_coords[0])
                f_y = int(HandStatus.finger_coords[1])
            except:
                f_x = bs.ball_pos[0]
                f_y = bs.ball_pos[1]
            bs.ball_velocity = np.array((bs.ball_pos[0] - f_x, bs.ball_pos[1] - f_y), dtype=float)
            bs.ball_velocity = bs.ball_velocity * 0.8  
        
        
        # Quit game
        if key == ord('q') or ENVIRONMENT.GamePoint >= 300: 
            break


    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()