tmux kill-session -t vr_bridge

# 创建一个名为"vr"的tmux会话
tmux new-session -d -s vr_bridge

tmux new-window -t vr_bridge:1
tmux send-keys -t vr_bridge:1 "unset ROS_DISTRO && source /opt/ros/noetic/local_setup.bash" C-m
tmux send-keys -t vr_bridge:1 "roscore" C-m

sleep 3s

# 创建一个新窗口（窗口1）用于Terminal B
tmux new-window -t vr_bridge:2
tmux send-keys -t vr_bridge:2 "unset ROS_DISTRO && source /opt/ros/noetic/local_setup.bash" C-m
tmux send-keys -t vr_bridge:2 "conda activate real-robo && python3 tools/teleoperation/vr_bridge.py" C-m

# 附加到tmux会话以便查看
tmux attach-session -t vr_bridge