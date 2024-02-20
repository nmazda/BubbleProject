import argparse
import cv2
from PIL import Image, ImageOps
from itertools import chain

import os
from pathlib import Path
import numpy as np
from typing import Iterator, List, Tuple

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mirror', help="flag for mirroring image", action='store_true', required=False, default=False)
    parser.add_argument('-s', '--split', help="flag for splitting image", action='store_true', required=False, default=False)
    parser.add_argument("input_path", help="path to input directory", type=Path)
    parser.add_argument("output_path", help="path to output directory", type=Path)
    return parser.parse_args()

def generate_dist_path(name: str) -> Path:
    name = name if name else "detect"
    for i in range(1000):
        dist_path = Path(f"runs/{name}{f'_{i}' if i > 0 else ''}")
        if not dist_path.exists():
            dist_path.mkdir(parents=True, exist_ok=True)
            return dist_path
    raise RuntimeError("Could not generate dist_path")

def mirror(input_img):
    img = Image.fromarray(input_img)
    return ImageOps.mirror(img)
    

def split(input_img, subsec):
    split = []
    # Creates image from np array
    img = Image.fromarray(input_img)

    for i in range(subsec):
        # Adds the subsections of height img.height/subsec to the split array
        split.append(img.crop((0, i * (img.height / subsec), img.width, (i + 1) * (img.height /subsec))))

    return split

def mirrorAndSplit(m, s, input_path, output_path):
    input_imgs = []

    if not m and not s: 
        print("No flag detected")
        return None

    # For all directories in the directory
    for dirpath, dirnames, filenames in os.walk(input_path):
        # For all files in the directory
        for filename in filenames:
            # If filename ends with .bmp, .jpg, or .png
            if filename.endswith('.bmp') or filename.endswith('.jpg') or filename.endswith('.png'):
                input_img = Image.open(f'{dirpath}/{filename}')
                output_img = []

                if (m): 
                    # Overwrights input image to mirror it
                    input_img = mirror(input_img)
            
                if (s): 
                    # Produces output image array of split pictuers
                    output_imgs = split(input_img)
                
                for i, image in enumerate(output_imgs):
                    image.save(f'{output_path}/{filename.stem}_{i}')


def main() -> None:
    args = get_args()
    mirrorAndSplit(
        args.mirror,
        args.split,
        args.input_path,
        args.output_path,
    )

if __name__ == "__main__":
    main()