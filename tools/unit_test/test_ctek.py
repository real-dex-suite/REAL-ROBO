            
import minimalmodbus
import serial
import threading
import time

# 寄存器地址
POSITION_HIGH_8 = 0x0102  # 位置寄存器高八位
POSITION_LOW_8 = 0x0103  # 位置寄存器低八位
SPEED = 0x0104
FORCE = 0x0105
MOTION_TRIGGER = 0x0108
BAUD = 115200
PORT = '/dev/ttyUSB0'  # 修改为您的COM口号


class Gripper:
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
            # self.instrument.read_long(POSITION_HIGH_8)
    # FORCE = 0x0105
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
    
    # open
    def open_gripper(self, width=0.1):
        self.write_position(8000) # 0 - 12cm 12000-->4000
        self.write_speed(80) # 0-100%
        # 写输入
        self.write_force(50)  # 
        # 触发运动
        self.trigger_motion()
    
    # close
    def close_gripper(self, force=42):
        self.write_position(12000) # 0 - 12cm 12000
        self.write_speed(80) # 0-100%
        # 写输入
        self.write_force(force)  # 
        # 触发运动
        self.trigger_motion()

    def gripper_digit(self,data):
        self.write_position(data) # 0 - 12cm 12000
        self.write_speed(80) # 0-100%
        # 写输入
        self.write_force(18)  # 
        # 触发运动
        self.trigger_motion()

if __name__ == '__main__':
    # 写入高八位位置
    # write_position_high8(instrument, 0x00)

    # 写入低八位位置
    # write_position_low8(instrument, 0x640)

    #写入位置

    gripper = Gripper()
    for i in range(20):
        gripper.open_gripper()
        print(gripper.read_force_state())
        print(gripper.read_position())
        time.sleep(3)
        print("**********")

     

        
        gripper.instrument.read_register(0x0612)

        gripper.gripper_digit(6000)

        print(gripper.read_force_state())
        print(gripper.read_position())
        time.sleep(3)
        gripper.gripper_digit(8000)
        print(gripper.read_force_state())
        print(gripper.read_position())
        time.sleep(3)
        gripper.close_gripper() # 0 - 12cm 12000
        print(gripper.read_force_state())
        print(gripper.read_position())
        time.sleep(3)
    # time.sleep(3)
    # gripper.open_gripper()