#!/bin/bash

# Default values for input and output directories
input_dir="results/tr1000e1000r10a01/test_latest/images"
output_dir="../SimToLabeledBubbleData/Real"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_dir) input_dir="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Create destination directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all files in the source directory
for filename in "$input_dir"/*_fake.png; do
    if [ -f "$filename" ]; then
        # Generate new filename
        new_filename=$(basename "$filename" _fake.png).png
        
        # Copy the file to destination directory with new name
        cp "$filename" "$output_dir/$new_filename"
        
        echo "Renamed and copied: $filename to $new_filename"
    fi
done
