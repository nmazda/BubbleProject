#!/bin/bash

RUNS_DIR="/home/iec/Documents/bubble_project/BubbleProject/bubble_detector/runs"
REAL_SPLIT_OUT="/home/iec/Documents/bubble_project/BubbleProject/local_copy/real_split"
BW_SPLIT_OUT="/home/iec/Documents/bubble_project/BubbleProject/local_copy/bw_split"

# TODO: Need to actually use the arguments
SRC_IMGS=$1
SPLT_AMNT=$2
CHPT=$3
REAL_SPLIT_OUT=$4
BW_SPLIT_OUT=$5

# Move to dir: bubble_detector
cd /home/iec/Documents/bubble_project/BubbleProject/bubble_detector

# Creates BW versions of Real images
python detect_BW.py ./models/bubble_swin-b/config.py $CHPT $SRC_IMGS
# Gets most recent detect folder
MOST_RECENT_DETECT=$(ls -td "$RUNS_DIR"/detect* 2>/dev/null | head -n 1)

# Move to dir: bubble_augmentation
cd /home/iec/Documents/bubble_project/BubbleProject/bubble_augmentation

if [ ! -d "$REAL_SPLIT_OUT" ]; then
    echo "$REAL_SPLIT_OUT does not exist. Creating it..."
    mkdir -p "$REAL_SPLIT_OUT"
fi

# Runs MirrorAndSplit with both mirroring and non mirroring for both real and bw
# Real Mirror then Non-Mirror
python ./mirrorAndSplit.py -m -s $SPLT_AMNT $SRC_IMGS $REAL_SPLIT_OUT
python ./mirrorAndSplit.py -s $SPLT_AMNT $SRC_IMGS $REAL_SPLIT_OUT

if [ ! -d "$BW_SPLIT_OUT" ]; then
    echo "$BW_SPLIT_OUT does not exist. Creating it..."
    mkdir -p "$BW_SPLIT_OUT"
fi

# BW Mirror then Non-Mirror
python ./mirrorAndSplit.py -m -s $SPLT_AMNT $MOST_RECENT_DETECT $BW_SPLIT_OUT
python ./mirrorAndSplit.py -s $SPLT_AMNT $MOST_RECENT_DETECT $BW_SPLIT_OUT
