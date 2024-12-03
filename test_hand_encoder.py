from holodex.components.robot_operators.robot import RobotController
import rospy
import time
import csv
import threading
from termcolor import cprint

def main():
    rospy.init_node("motor_accuracy_test")
    robot = RobotController(teleop=False)

    cmd_pos_1 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.57, 0., 0., 0.]
    # cmd_pos_2 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

    start_time = time.time()
    end_time = start_time + 30 * 60  # 30 mins

    csv_file = 'hand_positions.csv'
    # csv_headers = ['Timestamp (s)', 'Command Position', 'Actual Position', 'Reached Target']
    csv_headers = ['Timestamp (s)', 'Actual Position']
    robot.move_hand(cmd_pos_1)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

        while not rospy.is_shutdown():
            current_time = time.time()
            if current_time > end_time:
                break

            # Move to position 1
            # robot.move_hand(cmd_pos_1)
            # time.sleep(2)
            actual_pos = robot.get_hand_position()
            print(f"Command Position: {cmd_pos_1}, Actual Position: {actual_pos}")
            cprint(actual_pos, "green")
            # diff_1 = actual_pos[12] - cmd_pos_1[12]
            # cprint(diff_1, "cyan")
            writer.writerow([current_time - start_time, actual_pos])
            # time.sleep(4)

            # # Move to position 2
            # robot.move_hand(cmd_pos_2)
            # time.sleep(2)
            # actual_pos = robot.get_hand_position()
            # print(f"Command Position: {cmd_pos_2}, Actual Position: {actual_pos}")
            # diff_1 = actual_pos[12] - cmd_pos_2[12]
            # cprint(diff_1, "cyan")
            # writer.writerow([current_time - start_time, cmd_pos_2, actual_pos])

            # time.sleep(2)
            time.sleep(0.08)
            

    print(f"Data saved to {csv_file}")
    

if __name__ == "__main__":
    main()