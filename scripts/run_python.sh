#!/bin/bash

cd /home/jupyter/repos/steganalysis/
if test -f last-checkpoint.bin; then
    touch /tmp/startingfromcheckpoint.txt
    echo "Running from checkpoint"
    /opt/conda/bin/python baseline.py -c last-checkpoint.bin
else
    touch /tmp/nolastcheckpoint.txt
    echo "No last checkpoint, starting from scratch"
    /opt/conda/bin/python baseline.py
fi
sleep 20s