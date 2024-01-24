import argparse
from itertools import chain
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector
from PIL import Image

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
        "--score_thr", help="object confidence threshold for detection", type=float, default=0.4
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

    for image, image_path in DataLoader(source_path):
        result = inference_detector(model, image)
        #Calculates the bboxes and masks
        bboxes, masks = result

        #TODO: Quite a fair bit confused by this line, understand it
        bboxes = bboxes[0]  # [x1, y1, x2, y2, score]
        masks = np.array(masks[0])  # mask image of the same size as the original image

        # Remove masks where the bbox is 'unsure'/score < score_threshold
        masks = masks[np.where(bboxes[:, 4] > score_thr)]
        
        #TODO: Sums 2nd and 3rd axis of masks, unsure of what axis 0 is
        mask_areas = np.sum(masks, axis=(1, 2))
        
        #Prints the calculated min and max area of bubbles
        print(
            f"{image_path} {len(masks)} bubbles detected"
            f"(max area: {np.max(mask_areas)}[px^2], min area: {np.min(mask_areas)}[px^2])"
        )

        # Code below sourced from user PeterVennerstrom, altered to suite needs of current project
        # https://github.com/open-mmlab/mmdetection/issues/4713#issuecomment-879085909
        #
        # Creates binary B/W image of masks using fromArray from Pillow(Fork of PIL).
        # Saves mask_img to dist_path/image_path.name as jpeg, following the convention used in prior model.show result
        mask_img = [Image.fromarray(masks, mode='1')]
        [mask_img.save(f'{dist_path}/{image_path.name}.JPEG')]


        # model.show_result(
        #     image,
        #     result,
        #     out_file=dist_path / image_path.name,
        #     score_thr=score_thr,
        #     text_color=TEXT_COLOR,
        #     bbox_color=BBOX_COLOR,
        # )


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
