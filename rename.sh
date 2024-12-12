#!/bin/bash

# Default values for input and output directories
input_dir="C:\Users\Admin\Desktop\New\BubbleProject\SimToLabeledBubbleData\mergedBW"
output_dir="SimToLabeledBubbleData/Real"

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

# Check if any files match the pattern
shopt -s nullglob
files=("$input_dir"/*_fake.png)
if [ ${#files[@]} -eq 0 ]; then
    echo "No files found matching *_fake.png in $input_dir"
    exit 1
fi

# Loop through all files in the source directory
for filename in "${files[@]}"; do
    # Generate new filename
    new_filename=$(basename "$filename" _fake.png).png
    
    # Copy the file to destination directory with new name
    # -n prevents overwriting existing files
    cp -n "$filename" "$output_dir/$new_filename"
    
    echo "Renamed and copied: $filename to $new_filename"
done