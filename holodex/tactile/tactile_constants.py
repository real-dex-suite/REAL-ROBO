import numpy as np
from math import pi as PI

# Paxixni Tactile parameters
FORCE_LIMIT = 30
POINT_PER_SENSOR = 15
FORCE_DIM_PER_POINT = 3
PAXINI_FINGER_PART_NAMES = {
    'tip': 'cc',
    'pulp': 'aa'
}

BAUDRATE = 460800

# this decide the order of reading tactile for each sensor board
PAXINI_FINGER_PART_INFO = {
    'tip' : b'\xcc',
    'pulp' : b'\xaa'
}

PAXINI_GROUP_INFO = {
    0 : b'\xee',
    1 : b'\xff'
}

SERIAL_PORT_NUMBERS = ["/dev/ttyUSB0", "/dev/ttyUSB1"]

THUMB_TACTILE_INFO = {
    'serial_port_number': "/dev/ttyUSB0",
    'group_id': 0,
}
INDEX_TACTILE_INFO = {
    'serial_port_number': "/dev/ttyUSB0",
    'group_id': 1,
}
MIDDLE_TACTILE_INFO = {
    'serial_port_number': "/dev/ttyUSB1",
    'group_id': 0,
}
RING_TACTILE_INFO = {
    'serial_port_number': "/dev/ttyUSB1",
    'group_id': 1,
}

PAXINI_LEAPHAND = {
    "thumb": THUMB_TACTILE_INFO,
    "index": INDEX_TACTILE_INFO,
    "middle": MIDDLE_TACTILE_INFO,
    "ring": RING_TACTILE_INFO
}

TACTILE_FPS = 30

PAXINI_DP_ORI_COORDS = np.array([-4.70000,  3.30000, 2.94543,
                            -4.70000,  7.80000, 2.94543,
                            -4.70000, 12.30000, 2.94543,
                            -4.70000, 16.80000, 2.94543,
                            -4.70000, 21.30000, 2.94543,
                            0.00000,  3.30000, 3.09994,
                            0.00000,  7.80000, 3.09994,
                            0.00000, 12.30000, 3.09994,
                            0.00000, 16.80000, 3.09994,
                            0.00000, 21.30000, 3.09994,
                            4.70000,  3.30000, 2.94543,
                            4.70000,  7.80000, 2.94543,
                            4.70000, 12.30000, 2.94543,
                            4.70000, 16.80000, 2.94543,
                            4.70000, 21.30000, 2.94543]).reshape(-1,3)
PAXINI_IP_ORI_COORDS = np.array([-114.60000,  4.30109, 2.97814,
                            -114.60000,  8.15109, 2.89349,
                            -114.60000, 12.18660, 2.64440,
                            -114.60000, 15.99390, 2.06277,
                            -112.35000, 21.45300, 0.09510,
                            -110.10000,  4.30109, 3.10726,
                            -110.10000,  8.15109, 3.03111,
                            -110.10000, 12.20620, 2.80133,
                            -110.10000, 16.01800, 2.25633,
                            -110.10000, 24.50520,-2.49584,
                            -105.60000,  4.30109, 2.97814,
                            -105.60000,  8.15109, 2.89349,
                            -105.60000, 12.18660, 2.64440,
                            -105.60000, 15.99390, 2.06277,
                            -107.85000, 21.45300, 0.09510]).reshape(-1,3)

# for vis
PAXINI_DP_VIS_COORDS_2D = np.array([-4.70000,  3.30000, 3.09994,
                            -4.70000,  7.80000, 3.09994,
                            -4.70000, 12.30000, 3.09994,
                            -4.70000, 16.80000, 3.09994,
                            -4.70000, 21.30000, 3.09994,
                            0.00000,  3.30000, 3.09994,
                            0.00000,  7.80000, 3.09994,
                            0.00000, 12.30000, 3.09994,
                            0.00000, 16.80000, 3.09994,
                            0.00000, 21.30000, 3.09994,
                            4.70000,  3.30000, 3.09994,
                            4.70000,  7.80000, 3.09994,
                            4.70000, 12.30000, 3.09994,
                            4.70000, 16.80000, 3.09994,
                            4.70000, 21.30000, 3.09994]).reshape(-1,3)
PAXINI_DP_VIS_COORDS_2D -= np.mean(PAXINI_DP_VIS_COORDS_2D,0)
PAXINI_DP_VIS_COORDS_2D /= np.max(abs(PAXINI_DP_VIS_COORDS_2D))
PAXINI_DP_VIS_COORDS_2D /= 2

PAXINI_DP_VIS_COORDS_3D = np.array([-4.70000,  3.30000, 3.09994,
                            -4.70000,  7.80000, 3.09994,
                            -4.70000, 12.30000, 3.09994,
                            -4.70000, 16.80000, 3.09994,
                            -4.70000, 21.30000, 3.09994,
                            0.00000,  3.30000, 3.09994,
                            0.00000,  7.80000, 3.09994,
                            0.00000, 12.30000, 3.09994,
                            0.00000, 16.80000, 3.09994,
                            0.00000, 21.30000, 3.09994,
                            4.70000,  3.30000, 3.09994,
                            4.70000,  7.80000, 3.09994,
                            4.70000, 12.30000, 3.09994,
                            4.70000, 16.80000, 3.09994,
                            4.70000, 21.30000, 3.09994]).reshape(-1,3)

PAXINI_IP_VIS_COORDS_2D = np.array([[-4.5    ,  4.30109,  2.97814],
                            [-4.5    ,  8.15109,  2.89349],
                            [-4.5    , 12.1866 ,  2.6444 ],
                            [-4.5    , 15.9939 ,  2.06277],
                            [-2.25   , 21.453  ,  0.0951 ],
                            [ 0.     ,  4.30109,  3.10726],
                            [ 0.     ,  8.15109,  3.03111],
                            [ 0.     , 12.2062 ,  2.80133],
                            [ 0.     , 16.018  ,  2.25633],
                            [ 0.     , 24.5052 , -2.49584],
                            [ 4.5    ,  4.30109,  2.97814],
                            [ 4.5    ,  8.15109,  2.89349],
                            [ 4.5    , 12.1866 ,  2.6444 ],
                            [ 4.5    , 15.9939 ,  2.06277],
                            [ 2.25   , 21.453  ,  0.0951 ]])
PAXINI_IP_VIS_COORDS_2D -= np.mean(PAXINI_IP_VIS_COORDS_2D,0)
PAXINI_IP_VIS_COORDS_2D /= np.max(abs(PAXINI_IP_VIS_COORDS_2D))
PAXINI_IP_VIS_COORDS_2D /= 2

PAXINI_IP_VIS_COORDS_3D = np.array([[-4.5    ,  4.30109,  2.97814],
                            [-4.5    ,  8.15109,  2.89349],
                            [-4.5    , 12.1866 ,  2.6444 ],
                            [-4.5    , 15.9939 ,  2.06277],
                            [-2.25   , 21.453  ,  0.0951 ],
                            [ 0.     ,  4.30109,  3.10726],
                            [ 0.     ,  8.15109,  3.03111],
                            [ 0.     , 12.2062 ,  2.80133],
                            [ 0.     , 16.018  ,  2.25633],
                            [ 0.     , 24.5052 , -2.49584],
                            [ 4.5    ,  4.30109,  2.97814],
                            [ 4.5    ,  8.15109,  2.89349],
                            [ 4.5    , 12.1866 ,  2.6444 ],
                            [ 4.5    , 15.9939 ,  2.06277],
                            [ 2.25   , 21.453  ,  0.0951 ]])