# BubbleProject

## Overview ##

The Bubble Project has the overarching goal of making data collection/generation for a future neural network significantly easier. Current neural networks need a particularly large data set to train effectively, and in the case of a niche application such as evaluating the operations of a nuclear powerplant based on the bubbles it produces, getting this data is particularly difficult.

As such we have developed a pipeline for this purpose, working through the following principles. First, we gather a smaller data set of realistic bubble images and use the ONO app, built off of mmdetection, to segment the image into masked bubbles. These bubbles are then converted into a black and white 'mask' which will then be used to train a separate neural network model. This model takes these black and white images as input and is trained to reconstruct the realistic images. Doing so allows us to take an existing simulation program for these bubbles, take large sets of image data from them, convert them to black and white, and instruct this neural network to convert them into realistic images. The end result is a large data set of tagged realistic images that can easily be used to train a future neural network model.

## Black and White Conversion ##

The general mask/bounding box detection is done through MMdetection, as the ONO Detection app was built on the framework. However, this program does not support outputting just the masks to an image, as such, the following actions are done for each image. 

1. MMdetection is run, outputting the tuple "results" containing bboxes and masks.
2. Create an NP array for masks.
3. Removes masks below the provided threshold "score_thr".
5. Masks are combined across axis 0 into combined_mask, putting all bubble masks on the same plane.
7. Using PIL, combined_mask is transformed into mask_img using Image.fromArray.
8. mask_img is then converted to grey scale using .convert('L').
9. mask_img is saved to the detect-* folder under its original name, now converted to black and white. 

## Black and White Autoencoder ##

In order to create training data for a future neural network model, we train an autoencoder to take the created Black and White (B/W) images and convert them back into the original.
This way, when taking any data, such as the non-realistic simulation data, and turning it black and white, we can output a realistic image by running it through this neural network model.

## Setup of ONO app
[Here is the setup for the ONO app](https://github.com/nmazda/BubbleProject/blob/main/ONOSETUP.md)

## Overlap Detection

Overlap detection is used during the Ono detection in order to help reduce the number of overlapping bubbles in the training data for the autoencoder. The original implementation of the overlap detection would iteratively run through every pairing of bubbles and check their bounding boxes for overlap, and if any overlapped the image would be thrown out and no mask produced. This however led to a very small number of images being output, as most images had some form of overlap, or there was an issue with bubble hallucinations/non-detected bubbles, which led to this lack of output.

As such we have opted for a different approach. This new approach utilizes a reference image with a clear background to 'erase' the overlapping bubbles (Diagram Below)
![Brief diagram displaying how overlap detection currently works](https://github.com/nmazda/BubbleProject/blob/main/git_imgs/overlap_detection.jpg)

This allows for a significant increase in the number of images output and a much cleaner image than other techniques. However, in some cases, we feel there can still be additional bubbles kept. To do this we will take a heuristic approach to run through a group of overlapping bubbles from largest to smallest and keep the first that is still detected as a proper bubble by the detection program, likely being the largest bubble closest to the foreground. A diagram of this is below.
![Diagram showing the new heuristic approach to overlap detection](https://github.com/nmazda/BubbleProject/blob/main/git_imgs/overlap_heuristic.jpg)

## Mirror and Split data augmentation
[Here is how to use the data augmentation file]()



## Bash Script Usage
