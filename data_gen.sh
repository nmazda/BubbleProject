#!/bin/bash

RUNS_DIR="/home/iec/Documents/bubble_project/BubbleProject/bubble_detector/runs"

# TODO: Need to actually use the arguments
SRC_IMGS=$1
SPLIT_OUT=$2
SPLT_AMNT=$3
CHPT=$4

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
python ./mirrorAndSplit.py -m -s $SPLT_AMNT $SRC_IMGS $SPLIT_OUT/real_split
python ./mirrorAndSplit.py -s $SPLT_AMNT $SRC_IMGS $SPLIT_OUT/real_split

if [ ! -d "$BW_SPLIT_OUT" ]; then
    echo "$BW_SPLIT_OUT does not exist. Creating it..."
    mkdir -p "$BW_SPLIT_OUT"
fi

# BW Mirror then Non-Mirror
python ./mirrorAndSplit.py -m -s $SPLT_AMNT $MOST_RECENT_DETECT $SPLIT_OUT/bw_split
python ./mirrorAndSplit.py -s $SPLT_AMNT $MOST_RECENT_DETECT $SPLIT_OUT/bw_split
