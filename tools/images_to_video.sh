#!/bin/sh

FOLDER_PATH=$1
OUTPUT_PATH=$2

ffmpeg -framerate 8 -pattern_type glob -i "$FOLDER_PATH/*.jpg" -c:v libx264 -pix_fmt yuv420p $OUTPUT_PATH