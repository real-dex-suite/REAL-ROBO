import zmq
import rospy

def create_SUB_socket(HOST, PORT):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{HOST}:{PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    return socket

def process_received_message(message):
    #TODO
    print(f"Received message: {message}")


def publish_message2ROS(message):
    #TODO
    print(f"Publishing message: {message}")

def main(socket):
    while True:
        message = socket.recv_string()  # Receive message from the publisher
        print(f"Received message: {message}")

if __name__ == "__main__":
    rospy.init_node("get_vr_data", anonymous=True)
    socket = create_SUB_socket("172.21.11.178", 5555)
    main(socket)
