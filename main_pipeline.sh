#!/bin/bash

# Function to display a progress bar
show_progress() {
    local -r msg="$1"
    local -r pid="$2"
    local -r delay='0.75'
    local spinstr='\|/-'
    echo -n "$msg"
    while ps a | awk '{print $1}' | grep -q "$pid"; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
    echo ""
}

# Function to run a command with optional debug output
run_command() {
    local cmd="$1"
    local msg="$2"
    
    if [ "$DEBUG" = true ]; then
        echo "$msg"
        eval "$cmd"
    else
        eval "$cmd" > /dev/null 2>&1 &
        show_progress "$msg" $!
    fi

    # Check if the command failed
    if [ $? -ne 0 ]; then
        echo "Error: $msg failed."
        exit 1
    fi
}

# Parse command line arguments for debug mode
DEBUG=true
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG=true ;;
    esac
    shift
done

run_command "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate img2img-turbo" "Activating conda environment 'img2img-turbo'..."

run_command "python SimToLabeledBubbleData/get_bubble_info.py --input_dir SimToLabeledBubbleData/VOFdata --output_dir SimToLabeledBubbleData/Uncropped --json_output_dir SimToLabeledBubbleData/bubble_loc_data" "Running get_bubble_info.py..."

# run_command "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate opencv" "Activating conda environment 'opencv'..."

run_command "python SimToLabeledBubbleData/crop_resize.py --input_dir SimToLabeledBubbleData/Uncropped --output_dir SimToLabeledBubbleData/BW" "Running crop_resize.py..."

run_command "rsync -avz --progress SimToLabeledBubbleData/BW/ Pix2PixImageTranslation/datasets/BW/" "Copying BW folder to Pix2PixImageTranslation/datasets/..."

# run_command "conda activate pytorch-CycleGAN-and-pix2pix" "Activating conda environment 'pytorch-CycleGAN-and-pix2pix'..."

run_command "cd Pix2PixImageTranslation" "Changing directory to '../Pix2PixImageTranslation'..."

# Below command is heavy, and can't be changed mostly
run_command "cd Pix2PixImageTranslation" "Changing directory to '../Pix2PixImageTranslation'..."

run_command "python test.py --dataroot ./datasets/BW --name tr1000e1000r10a01 --model test --netG unet_256 --direction AtoB --dataset_mode single --norm batch --num_test 2000" "Running test.py..."

run_command "cd .." "Changing directory to 'BubbleProject"

run_command "./rename.sh --input_dir 'Pix2PixImageTranslation/results/tr1000e1000r10a01/test_latest/images' --output_dir 'SimToLabeledBubbleData/Real'" "Running rename.sh..."

run_command "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate opencv" "Activating conda environment 'opencv'..."

run_command "python SimToLabeledBubbleData/draw_bbox.py --input_dir 'SimToLabeledBubbleData/Real' --output_dir 'SimToLabeledBubbleData/LabelledBubbleData' --json_dir 'SimToLabeledBubbleData/bubble_loc_data'" "Running draw.py..."
