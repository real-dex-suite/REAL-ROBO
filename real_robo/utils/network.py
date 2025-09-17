import zmq
import time

def create_pull_socket(HOST, PORT):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind('tcp://{}:{}'.format(HOST, PORT))
    return socket

def create_push_socket(HOST, PORT):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind('tcp://{}:{}'.format(HOST, PORT))
    return socket

def frequency_timer(frequency):
    return time.sleep(1.0 / frequency)