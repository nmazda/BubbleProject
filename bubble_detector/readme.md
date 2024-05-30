# bubble detector

bubble detector is a model for detecting the position and shape of bubbles using mask cnn.

## Installation

Step 0.Download and install Maniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).
Step 1.Create a conda environment and activate it.
```bash
cd bubble_detector
conda create --name bubble_detector python=3.8 -y
conda activate bubble_detector
```
Step 2.Install Pytorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.
On GPU platforms:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
On CPU platforms:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Step 3. Install requirements
```bash
pip install -r requirements/pip.txt
mim install -r requirements/mim.txt
```

## Usage

Detecting
```bash
python detect.py ./models/bubble_swin-b/config.py ./models/bubble_swin-b/checkpoint.pth ./datasets/bubble_dataset/val/JPEGImages
```
Training
```bash
python train.py ./datasets/bubble_dataset
```
If you want to configure the learning settings detail, edit config/cascade_mask_rcnn__fpn.py following [official guide](https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/config.html).

## Included

bubble_dataset: Dataset for Instans Segmentation in COCO format with configuration in data.yaml.
bubblr_swin-b: Configuration and checkpoints learned mask-rcnn with swin-b backbone in bubble_dataset.
