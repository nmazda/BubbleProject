
# bubble detector

bubble detector is a model for detecting the position and shape of bubbles using mask cnn. This program can only be run on a GPU platform, as cuda capabilites are needed.

## Installation

Step 0.Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).
Step 1.Create a conda environment and activate it, terminal should be opened where you downloaded bubble_detector
```bash
cd bubble_detector
conda create --name bubble_project python=3.8 -y
conda activate bubble_project
```
Step 2.Install CudaToolKit
```bash
conda install anaconda::cudatoolkit==11.8.0
```
Step 3.Install Pytorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.
```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Step 4.Install OpenMIM
```bash
pip install -U openmim
```
Step 5.Install requirements
```bash
pip install -r requirements/pip.txt
mim install -r requirements/mim.txt
```

## Usage

Detecting Black and White, B/W images will be output in the runs folder under detect*
```bash
python detect_BW.py ./models/bubble_swin-b/config.py <path-to-checkpoint.pth <path-to-bubble-images>
```
Training
```bash
python train.py <path-to-training dataset>
```
If you want to configure the learning settings detail, edit config/cascade_mask_rcnn__fpn.py following [official guide](https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/config.html).

## Included

bubble_dataset: Dataset for Instans Segmentation in COCO format with configuration in data.yaml.
bubblr_swin-b: Configuration and checkpoints learned mask-rcnn with swin-b backbone in bubble_dataset.
