#!/bin/sh

# Check in the start of this reboot
touch /tmp/helloiamautocreatedtoday.txt
date > /tmp/helloiamautocreatedtoday.txt

# Create tmux session
tmux new -s training
cd /home/jupyter/repos/steganalysis/
if test -f last-checkpoint.bin; then
    python baseline.py -c last-checkpoint.bin
else
    echo "No last checkpoint, not doing anything"
fi