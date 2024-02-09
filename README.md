# BubbleProject

### Bubble Detection ###

Uses MMDetection's panoptic segmentation to mask individual bubbles within the input image.

### Reverse Black and White Autoencoder ###

In order to create training data for a future neural netork model, we train a autoencoder to take the created B/W images, and convert them back into the original.
This way, when taking any data, such as the non lifelike simulation data, and turning it black and white, we can out put a realistic image.

### Process of Black and White detection ###

The general mask/bounding box detection is done through mmdetection, as that was the original framework the ONO Detection app was built off of. However, this program does not support outputting just the mask as an immage, as such, the following actions are done for each image. 

1. mmdetection is run, outputting the tuple "results" containing bboxes and masks.
2. Creates an np array for masks and transfers data.
3. Removes masks below the provided threshold "score_thr".
4. Min and Max area are calculated for the bubbles, and text is output as such.
5. Masks are combined across axis 0 into combined_mask, putting all bubble masks on the same plane.
6. All values that are non-zero in the combined_mask array are set to 255. This creates the black and white setup.
7. Using PIL, combined_masks is transformed into mask_img using Image.fromArray.
8. mask_img is then fully converted to grey scale using .convert('L').
9. mask_img is saved to the detect-* folder under its original name, now converted to black and white. 

### Setup of ONO Detection app
[Here is the setup for the ONO Detection app](https://github.com/nmazda/BubbleProject/blob/main/ONOSETUP.md)
