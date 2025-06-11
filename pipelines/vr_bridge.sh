tmux kill-session -t vr_bridge

# 创建一个名为"vr"的tmux会话
tmux new-session -d -s vr_bridge

echo "========================"
echo ">>> Starting roscore"
echo "========================"
tmux send-keys -t vr_bridge:0 "unset ROS_DISTRO && source /opt/ros/noetic/local_setup.bash" C-m
tmux send-keys -t vr_bridge:0 "roscore" C-m

sleep 3s
tmux capture-pane -t vr_bridge:0 -p -S - -E - | grep -v "^$" | tail -n 10

# 创建一个新窗口（窗口1）用于Terminal B
echo "========================"
echo ">>> Starting vr_bridge"
echo "========================"
tmux new-window -t vr_bridge:1
tmux send-keys -t vr_bridge:1 "unset ROS_DISTRO && source /opt/ros/noetic/local_setup.bash" C-m
tmux send-keys -t vr_bridge:1 "conda activate real-robo && python3 tools/teleoperation/vr_bridge.py" C-m

sleep 3s
tmux capture-pane -t vr_bridge:1 -p -S - -E - | grep -v "^$" | tail -n 10
