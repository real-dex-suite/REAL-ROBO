from jkrc import jkrc
from holodex.constants import JAKA_IP

robot = jkrc.RC(JAKA_IP)#返回一个机器人对象
robot.login() #登录

robot.disable_robot() #关闭机器人
robot.power_off() #下电