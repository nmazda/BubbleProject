# Mirror and Split
## Usage
```bash
python ./MirrorAndSplit.py <-m> <-s [amount]> <input-dir> <output-dir>
```
### -m (Optional)
Mirror, include argument if you want output to be mirrored along the vertical axis.

### -s [amount] (Optional)
Split, splits the image vertically into amount sections. If given "-s 4" on a 256x1024 picture, the result would be 4 256x256 images.

### input-dir
The directory where the program can find the images you want to mirror and/or split.

### output-dir
The directory where you want the images that have been mirrored/split to be placed.




### Required Packages
- argparse
- PIL
- os
- pathlib
- numpy
- typing
