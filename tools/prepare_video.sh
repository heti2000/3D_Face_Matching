#!/bin/sh

VIDEO_PATH=$1
OUTPUT_PATH=$2
FPS=8

# Convert video to images
ffmpeg -i $VIDEO_PATH -vf fps=$FPS $OUTPUT_PATH/image_%04d.jpg

# run the landmark detector for each image
for f in $OUTPUT_PATH/*.jpg; do
    python3.11 tools/landmark_detector.py $OUTPUT_PATH $f
done
