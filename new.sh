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

run_command "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate C:\Users\Admin\anaconda03\envs\bubble_project" "Activating conda environment 'bubble_project'..."
run_command "python SimToLabeledBubbleData/get_bubble_info.py --input_dir SimToLabeledBubbleData/VOFdata --output_dir SimToLabeledBubbleData/Uncropped --json_output_dir SimToLabeledBubbleData/bubble_loc_data"
run_command "python SimToLabeledBubbleData/crop_resize.py --input_dir SimToLabeledBubbleData/Uncropped --output_dir SimToLabeledBubbleData/BW"
run_command "python SimToLabeledBubbleData/imagemerge.py"
run_command "python SimToLabeledBubbleData/find_bubble_border2.py --input_dir SimToLabeledBubbleData/mergedBW --output_dir SimToLabeledBubbleData/borderData"
run_command "rsync -avz --progress SimToLabeledBubbleData/BW/ Pix2PixImageTranslation/datasets/BW/" "Copying BW folder to Pix2PixImageTranslation/datasets/..."
run_command "cd Pix2PixImageTranslation" "Changing directory to '../Pix2PixImageTranslation'..."
run_command "d" "Running test.py..."
run_command "cd .." "Changing directory to 'BubbleProject"
run_command "./rename.sh --input_dir 'Pix2PixImageTranslation/results/tr1000e1000r10a01/test_latest/images' --output_dir 'SimToLabeledBubbleData/Real'" "Running rename.sh..."
# run_command "python SimToLabeledBubbleData/draw_bbox.py --input_dir 'SimToLabeledBubbleData/Real' --output_dir 'SimToLabeledBubbleData/LabelledBubbleData' --json_dir 'SimToLabeledBubbleData/bubble_loc_data"
# #Switch below code to run only edge files, not boxes
# run_command "python SimToLabeledBubbleData/draw_bbox.py --input_dir 'SimToLabeledBubbleData/Real' --output_dir 'SimToLabeledBubbleData/LabelledBubbleData' --json_dir 'SimToLabeledBubbleData/bubble_loc_data'" "Running draw.py..."


#Store final data as realistic images and their corresponding json files in seperate folders