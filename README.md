# BubbleProject

### Bubble Detection ###

Uses MMDetection's panoptic segmentation to mask individual bubbles within the input image.

### Reverse Black and White Autoencoder ###

In order to create training data for a future neural netork model, we train a autoencoder to take the created B/W images, and convert them back into the original.
This way, when taking any data, such as the non lifelike simulation data, and turning it black and white, we can out put a realistic image.
