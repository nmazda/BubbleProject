# Simulation To Labeled Bubble Data

This repository contains scripts for processing images in following main steps to generated labeled bubble data from Simulation images: 
1) Read simulation data from .dat files and visualize
2) Converting original images to resized squares
3) Generating realistic images using a Pix2Pix model 
4) Draw bounding boxes uysing bubble detection location data saved in JSON files
5) [Optional] Combining paired images. Follow the instructions below to execute each step.

## Step 1: Read simulation data from .dat files and visualize

This project involves visualizing data from bubble simulations. The data is stored in 77 .dat files, each corresponding to a snapshot of the simulation. Each file contains a large binary dataset that represents a 3D grid of values. The main goal of the project is to identify and isolate bubble locations within this 3D space and then generate black-and-white images from the Y-Z perspective.

#### Data Description
Each .dat file contains 204,800,000 bits of data, saved as uint32 dtype binary format. When read and reshaped, each file represents an 80x80x1000 3D grid:

Dimensions: 80 (X) x 80 (Y) x 1000 (Z)
Data Type: uint32

<img src="https://github.com/nmazda/BubbleProject/blob/main/git_imgs/sim_data_format.png" width="600" height="400">

#### Create conda env from yaml file 
Download the env.yaml
then execute following command to create env
```bash
conda env create --file bubble_project.yaml --name bubble_project
```

#### Execute following command to install required libraries

First Download and install python 3.9

```bash
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install vtk
pip install pyside2
pip install mayavi
pip install configobj

```

#### Execute following command to visulize the simulation data
```bash
python read_sim_data_visualize.py --input_dir path/to/input 
```
Note: If you want to save the visualization as image use Mayavi scene toolbar at top to save 

<img src="https://github.com/nmazda/BubbleProject/blob/main/git_imgs/MayaviScene.png" width="550" height="600">

#### Execute following command to detect and save bubbles location information from the simulation data
```bash
python get_bubble_info.py --input_dir VOFdata --output_dir Uncropped --json_output_dir bubble_loc_data
```

## Step 2: Converting Original Images to Resized Squares

The first step involves converting original images of size 1612x786 into squares of 60x60 pixels, then resizing these squares to 256x256 pixels, and saving them to a directory.

#### Execute following command to generate B&W images
```bash
python SimToLabeledBubbleData/crop_resize.py --input_dir SimToLabeledBubbleData/Uncropped --output_dir SimToLabeledBubbleData/BW
# copy images from SimToLabeledBubbleData to Pix2PixImageTranslation/datasets
rsync -avz --progress SimToLabeledBubbleData/BW/ Pix2PixImageTranslation/datasets/BW/
```

<img src="https://github.com/nmazda/BubbleProject/blob/main/git_imgs/original_img_to_bw_sqrs.png" width="600" height="400">


## Step 3: Generating Realistic Images using Pix2Pix Model
Using the prepared black and white (B&W) images, this step generates realistic images using a trained Pix2Pix model. The generated realistic images and the original B&W images are saved to a specified directory.

#### Execute following command to generate realistic images from B&W images
```bash
python test.py --dataroot path/to/dataset --name model_name --model test --netG unet_256 --direction AtoB --dataset_mode single --norm batch --num_test <no of images>
```
Replace path/to/dataset with the path to your dataset, and model_name with the name of your trained Pix2Pix model. This script uses the specified model to generate realistic images from the B&W images.


<img src="https://github.com/nmazda/BubbleProject/blob/main/git_imgs/bw_to_realistic_img.png" width="600" height="400">

#### Execute following command to rename and copy generated realistic images to correct directory
```bash
./rename.sh --input_dir 'results/tr1000e1000r10a01/test_latest/images' --outputdir '../SimToLabeledBubbleData/Real'
```


## Step 4: Draw Bounding Boxes
In the final step, this script draw bounding boxes using the bubble detection location data from JSON files

#### Execute following command to combine images and save single paired image
```bash
python SimToLabeledBubbleData/draw_bbox.py --input_dir 'SimToLabeledBubbleData/Real' --output_dir 'SimToLabeledBubbleData/LabelledBubbleData' --json_dir 'SimToLabeledBubbleData/bubble_loc_data'
```
This scripts generated labelled data of bubble detection.


<img src="https://github.com/nmazda/BubbleProject/blob/main/git_imgs/fl_0_0450_10_xz.png" >


## [Optional]: Combining Paired Images
In the final step, this script combines paired B&W and realistic images side by side and saves the combined images to a new directory.

#### Execute following command to combine images and save single paired image
```bash
python combine_paired_images.py --input_dir path/to/input --output_dir path/to/output
```
This script processes images in the paired_images directory and saves the combined images to the combined_images directory.


<img src="https://github.com/nmazda/BubbleProject/blob/main/git_imgs/combine_paired_imgs.png" width="600" height="400">
