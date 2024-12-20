# BubbleProject

## Overview ##

The Bubble Project has the overarching goal of making data collection/generation for a future neural network significantly easier. Current neural networks need a particularly large data set to train effectively, and in the case of a niche application such as evaluating the operations of a nuclear powerplant based on the bubbles it produces, getting this data is particularly difficult.

As such we have developed a pipeline for this purpose, working through the following principles. First, we gather a smaller data set of realistic bubble images and use the ONO app, built off of mmdetection, to segment the image into masked bubbles. These bubbles are then converted into a black and white 'mask' which will then be used to train a separate neural network model. This model takes these black and white images as input and is trained to reconstruct the realistic images. Doing so allows us to take an existing simulation program for these bubbles, take large sets of image data from them, convert them to black and white, and instruct this neural network to convert them into realistic images. The end result is a large data set of tagged realistic images that can easily be used to train a future neural network model.

## GAN based Image Translation ##

We utilize a Generative Adversarial Network (GAN) called Pix2Pix for translating black-and-white bubble images into realistic grayscale images. Pix2Pix is a popular image-to-image translation framework that uses paired images to learn the mapping between input and output images. This process helps us generate realistic grayscale images from simulation data, which can then be used to create a large dataset for training future neural networks.

### Pix2Pix

The Pix2Pix architecture consists of a generator that creates realistic images from input data and a discriminator that evaluates their authenticity. During adversarial training, the generator and discriminator are optimized together, improving the generator's ability to produce high-quality, realistic images. For more details please refer original [paper](https://arxiv.org/pdf/1611.07004).

#### Dataset Preparation

To prepare our dataset for Pix2Pix training, we organized our data into paired images {A, B} representing different depictions of the same scene. These pairs were stored in folders /path/to/data/A and /path/to/data/B, with corresponding subdirectories for different data splits. After ensuring that images in each pair had the same size and filename, we used a Python script to combine each pair into a single image file, ready for training. For training, input images are presented as paired images with dimensions of 256x256 pixels. Prior to training, we perform preprocessing steps on these input images, including resizing and cropping.

#### Implementation and Output

In this project, we utilize PyTorch, a powerful deep learning framework. To delve deeper into the specifics of our training and testing methodologies, please refer to our documentation [here](Pix2PixImageTranslation/README.md). The output of our model comprises realistic bubble images with dimensions of 256x256 pixels. For additional insights into the Pix2Pix model, we recommend referring to the original [PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository detailing its implementation.

### Pipeline To Convert Simulation Images to Realistic Labeled Bubble Data

This pipeline processes simulation images to generate labeled bubble data through a series of steps, including image conversion, realistic image generation using the Pix2Pix GAN image translation model, image combination, and labelling the bubbles in realistic images using labels from simulation. Initially, it converts original simulation images into smaller black and white (B&W) squares, resizing them to a standard size. Using a trained Pix2Pix model, it then generates realistic images from these B&W squares. The pipeline combines each pair of B&W and realistic images side by side, producing paired images. Finally, it applies the bubble labels from the simulation data to the generated realistic images, resulting in a fully labeled bubble dataset.

Please refer to our documentation [here](https://github.com/nmazda/BubbleProject/blob/main/SimToLabeledBubbleData/README.md) for more details about the steps to execute the scripts.

#### To run main pipeline to generate labelled bubble detection data from .dat files
Run following command (Old)
```bash
chmod +x main_pipeline.sh
./main_pipeline.sh
```

Run following command (latest)
```bash
chmod +x new.sh
./new.sh
```

All the results, generated images and weights are saved at [one drive](https://1drv.ms/f/s!ArZaiTbmszajgbGPKLKluNillG28wps?e=tyg7xk)

## Black and White Bubble Data Conversion ##

Black and white conversion is necessary as it provides an easy middle ground for both the original training data of the autoencoder as well as the simulation data. Having this middle ground allows us to train the neural network to reconstruct the realistic data from the converted black-and-white realistic images and still be able to apply it to the simulation data once it's converted to black-and-white.

The general mask/bounding box detection is done through MMdetection, as the ONO Detection app was built on the framework. However, this program does not support outputting just the masks to an image, as such, the following actions are done for each image. 

1. MMdetection is run, outputting the tuple "results" containing bboxes and masks.
2. Create an NP array for masks.
3. Removes masks below the provided threshold "score_thr".
5. Masks are combined across axis 0 into combined_mask, putting all bubble masks on the same plane.
7. Using PIL, combined_mask is transformed into mask_img using Image.fromArray.
8. mask_img is then converted to grey scale using .convert('L').
9. mask_img is saved to the detect-* folder under its original name, now converted to black and white. 

## Mirror and Split data augmentation
[Here is how to use the data augmentation file]()

## Setup of ONO app
[Here is the setup for the ONO app](https://github.com/nmazda/BubbleProject/blob/main/ONOSETUP.md)

## Overlap Detection (Not used in final pipeline)

Overlap detection was explored for use during the Ono detection in order to help reduce the number of overlapping bubbles in the training data for the autoencoder. The original implementation of the overlap detection would iteratively run through every pairing of bubbles and check their bounding boxes for overlap, and if any overlapped the image would be thrown out and no mask produced. This however led to a very small number of images being output, as most images had some form of overlap, or there was an issue with bubble hallucinations/non-detected bubbles, which led to this lack of output.

As such we explored a different approach. This new approach utilizes a reference image with a clear background to 'erase' the overlapping bubbles (Diagram Below)
![Brief diagram displaying how overlap detection currently works](https://github.com/nmazda/BubbleProject/blob/main/git_imgs/overlap_detection.jpg)

This allows for a significant increase in the number of images output and a much cleaner image than other techniques. However, in some cases, we feel there can still be additional bubbles kept. To do this we will take a heuristic approach to run through a group of overlapping bubbles from largest to smallest and keep the first that is still detected as a proper bubble by the detection program, likely being the largest bubble closest to the foreground. A diagram of this is below.
![Diagram showing the new heuristic approach to overlap detection](https://github.com/nmazda/BubbleProject/blob/main/git_imgs/overlap_heuristic.jpg)

## Bash Script Usage
The bash script will output two folders, bw_split and real_split to the path_to_output_directories directory. 
```
./data_gen.sh [path_to_input_images] [path_to_output_directories] [number_of_vertical_splits_in_output_image] [path_to_checkpoint_file]
```
