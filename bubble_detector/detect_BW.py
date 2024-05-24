import argparse
from itertools import chain
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector
from PIL import Image, ImageOps
import matplotlib.pyplot as plt




TEXT_COLOR = (255, 255, 255)
BBOX_COLOR = (72, 101, 241)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="path to config file", type=Path)
    parser.add_argument("checkpoint_path", help="path to checkpoint path", type=Path)
    parser.add_argument(
        "source_path", help="path to source directory or file for image", type=Path
    )
    parser.add_argument("--name", help="experiment name", type=str, default="")
    parser.add_argument(
        "--score_thr", help="object confidence threshold for detection", type=float, default=0.94
    )
    parser.add_argument(
        "--device",
        help="device to run on, i.e. cuda:0/1/2/3 or device=cpu",
        type=str,
        default="cuda:0",
    )
    return parser.parse_args()


def generate_dist_path(name: str) -> Path:
    name = name if name else "detect"
    for i in range(1000):
        dist_path = Path(f"runs/{name}{f'_{i}' if i > 0 else ''}")
        if not dist_path.exists():
            dist_path.mkdir(parents=True, exist_ok=True)
            return dist_path
    raise RuntimeError("Could not generate dist_path")


class DataLoader(object):
    IMG_EXTS = ["bmp", "jpg", "png"]

    def __init__(self, source_path: Path) -> None:
        self._image_paths = self._get_image_paths(source_path)
        self._i = 0

    def __len__(self) -> int:
        return len(self._image_paths)

    def __iter__(self) -> Iterator:
        self._i = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, Path]:
        if self._i >= len(self):
            raise StopIteration()
        image_path = self._image_paths[self._i]
        image = np.array(Image.open(image_path))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        self._i += 1
        return image, image_path

    @classmethod
    def _get_image_paths(cls, source_path: Path) -> List[Path]:
        if source_path.is_dir():
            image_paths = chain.from_iterable(
                source_path.glob(f"**/*.{ext}") for ext in cls.IMG_EXTS
            )
            return list(image_paths)
        elif source_path.is_file() and source_path.suffix in cls.IMG_EXTS:
            return [source_path]
        else:
            raise Exception(f"Invalid source path: {source_path}")

def isOverlapping(bbox1, bbox2) -> bool:
    # Pixel/Screen Coordinates are apparently backwards in terms of +/- Y, where +Y goes lower on the screen.
    # As such, B1 Top, means the "botto"m of the square, and B1 Bottom means "Top" of the square when imagining in cartesian plane
    #            B1 Left >= B2 Right     B1 Right <= B2 Left    B1 Top >= B2 Bottom     B1 Bottom <= B2 Top 
    # print(bbox1)
    # print(bbox2)
    return not (bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1])

def detect(
    config_path: Path,
    checkpoint_path: Path,
    source_path: Path,
    name: str,
    score_thr: float,
    device: str,
) -> None:
    model = init_detector(str(config_path), str(checkpoint_path), device=device)
    dist_path = generate_dist_path(name)

    # Loads ref background img as img
    ref_background_arr = np.array(Image.open("/home/iec/Documents/bubble_project/BubbleProject/bubble_detector/req_files/Background.jpg"))
    ref_background_arr = cv2.cvtColor(ref_background_arr, cv2.COLOR_RGBA2RGB)

    k = 0
    for image, image_path in DataLoader(source_path):
        result = inference_detector(model, image)
        #Calculates the bboxes and masks
        bboxes, masks = result

        bboxes = bboxes[0]  # [x1, y1, x2, y2, score]
        masks = np.array(masks[0])  # mask image of the same size as the original image

        # Remove masks & bboxes where the bbox is 'unsure'/score < score_threshold
        masks = masks[np.where(bboxes[:, 4] > score_thr)]
        bboxes = [bbox for bbox in bboxes if bbox[4] > score_thr]

        if masks.size == 0:
            k = k + 1
            print(f"No Masks Found: Image {k}")
            continue

        #Sums total area of bubbles in image.
        mask_areas = np.sum(masks, axis=(1, 2))   

        overlap_idxs = set()    

        # # Iterates through all bboxes and checks if theyre overlapping, if overlapping, skip the image saving process.
        for i, bbox1 in enumerate(bboxes):
            for j, bbox2 in enumerate(bboxes):
                if i != j and isOverlapping(bbox1, bbox2):
                    overlap_idxs.add(i)
                    overlap_idxs.add(j)
                    
        

        # Prints the calculated min and max area of bubbles
        print(
            f"{image_path} {len(masks)} bubbles detected"
            f"(max area: {np.max(mask_areas)}[px^2], min area: {np.min(mask_areas)}[px^2])"
        )

        model.show_result(
            image,
            result,
            out_file=dist_path / image_path.name,
            score_thr=score_thr,
            text_color=TEXT_COLOR,
            bbox_color=BBOX_COLOR,
        )

        for idx in overlap_idxs:
            # image[masks[idx] > 0] = 0
            image[masks[idx] > 0] = ref_background_arr[masks[idx] > 0]

        #Combines all masks into one np array
        combined_mask = np.sum(masks, axis=0).astype(np.uint8)

        #Replaces all values non 0 in np array with 255.
        combined_mask[combined_mask > 0 ] = 255

        #Creates an image from the combined_mask np array.
        mask_img = Image.fromarray(combined_mask)
        org_img = Image.fromarray(image)

        #Converts mask_img to greyscale image.
        mask_img = mask_img.convert('L')

        #Invert Image
        mask_img = ImageOps.invert(mask_img)

        #Saves mask_img to runs folder under its original name.
        # mask_img.save(f'{dist_path}/{image_path.stem}.jpg')
        org_img.save(f'{dist_path}/{image_path.stem}*.jpg')



def main() -> None:
    args = get_args()
    detect(
        args.config_path,
        args.checkpoint_path,
        args.source_path,
        args.name,
        args.score_thr,
        args.device,
    )


if __name__ == "__main__":
    main()
