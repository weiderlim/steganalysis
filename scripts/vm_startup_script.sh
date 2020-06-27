#!/bin/sh

# Check in the start of this reboot
touch /tmp/helloiamautocreatedtoday.txt
date > /tmp/helloiamautocreatedtoday.txt

# Create tmux session
tmux new-session -d -s training /home/jupyter/repos/steganalysis/scripts/run_python.sh