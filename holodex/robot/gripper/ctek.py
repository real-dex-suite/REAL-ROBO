            
import minimalmodbus
import threading
import time

# 寄存器地址
POSITION_HIGH_8 = 0x0102  # 位置寄存器高八位
POSITION_LOW_8 = 0x0103  # 位置寄存器低八位
SPEED = 0x0104 # 0 -- 100
FORCE = 0x0105 # 0 -- 40
MOTION_TRIGGER = 0x0108
BAUD = 115200
PORT = '/dev/ttyUSB0'  # 修改为您的COM口号


class CTekGripper:
    """gripper init"""

    def __init__(self):
        self.instrument = minimalmodbus.Instrument(PORT, 1)
        self.instrument.serial.baudrate = BAUD
        self.instrument.serial.timeout = 1
        self.lock = threading.Lock()

    # 写入高八位位置
    def write_position_high8(self, value):
        with self.lock:
            self.instrument.write_register(POSITION_HIGH_8, value, functioncode=6)

    # 写入低八位位置
    def write_position_low8(self, value):
        with self.lock:
            self.instrument.write_register(POSITION_LOW_8, value, functioncode=6)

    # 写入位置
    def write_position(self, value):
        with self.lock:
            self.instrument.write_long(POSITION_HIGH_8, value)

    def read_position(self):
            return self.instrument.read_register(0x060A)
    
    def read_force_state(self):
            return self.instrument.read_register(0x060C)

    # 写入速度
    def write_speed(self, speed):
        with self.lock:
            self.instrument.write_register(SPEED, speed, functioncode=6)

    # 写输入
    def write_force(self, force):
        with self.lock:
            self.instrument.write_register(FORCE, force, functioncode=6)

    # 触发运动
    def trigger_motion(self):
        with self.lock:
            self.instrument.write_register(MOTION_TRIGGER, 1, functioncode=6)
    
    # 开爪
    def open_gripper(self, speed=80, force=40, block=True, waiting_time=3):
        self.write_position(4000) # 0 - 12cm 12000-->4000
        self.write_speed(speed)
        # 写输入
        self.write_force(force)
        # 触发运动
        self.trigger_motion()
        if block:
            time.sleep(waiting_time)
    
    # 关爪
    def close_gripper(self, speed=80, force=40, block=True, waiting_time=3):
        self.write_position(12000)
        self.write_speed(speed)
        # 写输入
        self.write_force(force)
        # 触发运动
        self.trigger_motion()
        if block:
            time.sleep(waiting_time)

    def move_gripper(self, position, speed=80, force=40, block=True, waiting_time=3):
        self.write_position(position)
        self.write_speed(speed)
        # 写输入
        self.write_force(force)
        # 触发运动
        self.trigger_motion()
        if block:
            time.sleep(waiting_time)

if __name__ == '__main__':
    gripper = CTekGripper()

    gripper.close_gripper()
    time.sleep(3)
    print(">>>> opening")
    gripper.open_gripper()
    cnt = 0
    while cnt <= 3:
        print(f"t: {cnt:.1f}, force_state: {gripper.read_force_state()}, position: {gripper.read_position()}")
        time.sleep(0.1)
        cnt += 0.1
    print("============")
    print(">>>> closing")
    gripper.close_gripper()
    cnt = 0
    while cnt <= 3:
        print(f"t: {cnt:.1f}, force_state: {gripper.read_force_state()}, position: {gripper.read_position()}")
        time.sleep(0.1)
        cnt += 0.1
    print("============")

    # print(">>> close 0cm")
    # gripper.close_gripper()
    # time.sleep(3)
    # print("force_state:", gripper.read_force_state())
    # print("position:", gripper.read_position())
    # print(">>>>>>>>")
    # print(">>> open 8cm")
    # gripper.open_gripper()
    # time.sleep(3)
    # print("force_state:", gripper.read_force_state())
    # print("position:", gripper.read_position())
    # print(">>>>>>>>")

    # print(">>> move 6cm")
    # gripper.move_gripper(6000)
    # time.sleep(3)
    # print("force_state:", gripper.read_force_state())
    # print("position:", gripper.read_position())
    # print(">>>>>>>>")

    # print(">>> move 4cm")
    # gripper.move_gripper(8000)
    # time.sleep(3)
    # print("force_state:", gripper.read_force_state())
    # print("position:", gripper.read_position())
    # print(">>>>>>>>")

    # print(">>> move 2cm")
    # gripper.move_gripper(10000)
    # time.sleep(3)
    # print("force_state:", gripper.read_force_state())
    # print("position:", gripper.read_position())
    # print(">>>>>>>>")

    # print(">>> move 0cm")
    # gripper.move_gripper(12000)
    # time.sleep(3)
    # print("force_state:", gripper.read_force_state())
    # print("position:", gripper.read_position())
    # print(">>>>>>>>")

    # print(">>> move 8cm")
    # gripper.move_gripper(4000)
    # time.sleep(3)
    # print("force_state:", gripper.read_force_state())
    # print("position:", gripper.read_position())
    # print(">>>>>>>>")