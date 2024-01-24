import pyrealsense2 as rs

ctx = rs.context()
devs = ctx.query_devices() # devs是device_list类
device_num = devs.size()

print(device_num)
# 1
print(len(devs))
# 1

for dev in devs:
    print(type(dev))
    # <class 'pyrealsense2.pyrealsense2.device'>

    serial_number = dev.get_info(rs.camera_info.serial_number)
    print(serial_number)
    # 838212074152