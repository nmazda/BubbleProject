import argparse
from PIL import Image, ImageOps

import os
from pathlib import Path
import numpy as np
from typing import Iterator, List, Tuple

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mirror', help="flag for mirroring image", action='store_true', required=False, default=False)
    parser.add_argument('-s', '--split', help="flag for splitting image", required=False, type=int, default=0)
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
    print("Mirror")
    return ImageOps.mirror(input_img)
    

def split(input_img, subsec):
    print("Split")
    split = []
    # Creates image from np array

    for i in range(subsec):
        # Adds the subsections of height img.height/subsec to the split array
        split.append(input_img.crop((0, i * (input_img.height / subsec), input_img.width, (i + 1) * (input_img.height /subsec))))

    return split

def mirrorAndSplit(m, s, input_path, output_path):
    if (not m and s <= 0): 
        print("No flag detected")
        return None

    # For all directories in the directory
    for dirpath, dirnames, filenames in os.walk(input_path):
        # For all files in the directory
        for filename in filenames:
            # If filename ends with .bmp, .jpg, or .png
            if filename.endswith('.bmp') or filename.endswith('.jpg') or filename.endswith('.png'):
                input_img = Image.open(f'{dirpath}/{filename}')
                output_imgs = []

                # Mirror true, split non zero
                if (m and s):
                    input_img = mirror(input_img)
                    output_imgs = split(input_img, s)

                #Mirror true
                elif (m):
                    output_imgs.append(mirror(input_img))

                #Split non zero
                elif (s):
                    output_imgs = split(input_img, s)

                
                #For each image(Accounting for the multiple images created by split) save
                for i, image in enumerate(output_imgs):
                    filestem = os.path.splitext(filename)[0]

                    #if image was mirrored, add a * to the filename
                    mirrored = ""
                    if m: mirrored = "*"
                    
                    #outputdir/fimestem_0*.jpg
                    image.save(f'{output_path}/{filestem}_{i}{mirrored}.jpg')


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