# Simulation To Labeled Bubble Data

This repository contains scripts for processing images in three main steps: converting original images to resized squares, generating realistic images using a Pix2Pix model, and combining paired images. Follow the instructions below to execute each step.

## Step 1: Converting Original Images to Resized Squares

The first step involves converting original images of size 1612x786 into squares of 60x60 pixels, then resizing these squares to 256x256 pixels, and saving them to a directory.

##### Execute following command to generate B&W images
```bash
python prepare_bw_images.py
```

## Step 2: Generating Realistic Images using Pix2Pix Model
Using the prepared black and white (B&W) images, this step generates realistic images using a trained Pix2Pix model. The generated realistic images and the original B&W images are saved to a specified directory.

##### Execute following command to generate realistic images from B&W images
```bash
python test.py --dataroot path/to/dataset --name model_name --model test --netG unet_256 --direction AtoB --dataset_mode single --norm batch
```
Replace path/to/dataset with the path to your dataset, and model_name with the name of your trained Pix2Pix model. This script uses the specified model to generate realistic images from the B&W images.

## Step 3: Combining Paired Images
In the final step, this script combines paired B&W and realistic images side by side and saves the combined images to a new directory.

##### Execute following command to combine images and save single paired image
```bash
python combine_paired_images.py
```
This script processes images in the paired_images directory and saves the combined images to the combined_images directory.
