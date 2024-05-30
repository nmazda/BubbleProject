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
    parser.add_argument('--overlap_det', dest='overlap_det', action='store_true')
    parser.add_argument('--ono_out', dest='ono_out', action='store_true')
    parser.add_argument("config_path", help="path to config file", type=Path)
    parser.add_argument("checkpoint_path", help="path to checkpoint path", type=Path)
    parser.add_argument(
        "source_path", help="path to source directory or file for image", type=Path
    )
    parser.add_argument("--name", help="experiment name", type=str, default="")
    parser.add_argument(
        "--score_thr", help="object confidence threshold for detection", type=float, default=0.96
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

# Returns true if the two bboxes overlap
def isOverlapping(bbox1, bbox2) -> bool:
    # Pixel/Screen Coordinates are apparently backwards in terms of +/- Y, where +Y goes lower on the screen.
    # As such, B1 Top, means the "botto"m of the square, and B1 Bottom means "Top" of the square when imagining in cartesian plane
    #            B1 Left >= B2 Right     B1 Right <= B2 Left    B1 Top >= B2 Bottom     B1 Bottom <= B2 Top 
    return not (bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1])

# Returns the area of a bbox
def getArea(bbox) -> int:
    # Bbox area = (right - left) * (top - bottom)
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

# Builds a graph of bubble bboxes
def build_graph(bboxes):
    n = len(bboxes)
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if isOverlapping(bboxes[i], bboxes[j]):
                graph[i].append(j)
                graph[j].append(i)
    return graph

# Finds all connected 'nodes' in the 'graph' of overlapping bboxes
def find_connected_components(graph):
    visited = set()
    components = []
    
    def dfs(node, component):
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
    
    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
    
    return components

def overlapDetection(masks, bboxes, image, ref_background_arr, model):
    # overlap_idxs = set()    

    # # # Iterates through all bboxes and checks if theyre overlapping, if overlapping, skip the image saving process.
    # for i, bbox1 in enumerate(bboxes):
    #     for j, bbox2 in enumerate(bboxes):
    #         if i != j and isOverlapping(bbox1, bbox2):
    #             if (getArea(bbox1) > getArea(bbox2)):
    #                 overlap_idxs.add(j)
    #             else:
    #                 overlap_idxs.add(i)

    bubble_graph = build_graph(bboxes)
    overlap_bubbles = find_connected_components(bubble_graph)
    print(overlap_bubbles)

    non_overlap_org_img = np.copy(image)

    # Decides which bubbles from each group to keep based on which is largest(by bbox)
    overlap_bubbles_keep = set()
    for group in overlap_bubbles:
        largest_bubble_idx = 0
        maxArea = getArea(bboxes[0])

        for idx in group:
            if (getArea(bboxes[idx]) > maxArea):
                maxArea = getArea(bboxes[idx])
                largest_bubble_idx = idx
        overlap_bubbles_keep.add(largest_bubble_idx)
            
    # # Transforms graph to set form
    # overlap_bubbles_remove = set()
    # for node, neighbors in overlap_bubbles.items():
    #     if len(neighbors) > 1:
    #         overlap_bubbles_remove.add(node)
    #         overlap_bubbles_remove.update(neighbors)

    # Removes bubbles in overlap_bubbles_remove which arent in overlap_bubbles_keep
    for idx in masks:
        if not idx in overlap_bubbles_keep:
            binary_mask = masks[idx].astype(np.uint8)

            # Kernel used to "blur" or "expand" the range of the bubble to cover the edges.
            kernel = np.ones((5, 5), np.uint8)

            # Dialated Mask for where the bubble is.
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            # Replace detected overlaps with the respective area in the background.
            non_overlap_org_img[dilated_mask > 0] = ref_background_arr[dilated_mask > 0]        

    # Removes Overlaping Idxs from masks and bboxes
    non_overlap_indices = [i for i in range(len(masks)) if i not in overlap_bubbles_keep]
    masks = masks[non_overlap_indices]

    return non_overlap_org_img

def detect(
    overlap_det: bool,
    ono_out: bool,
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
            print(f"No Masks Found: Image {k}")
            continue

        if (ono_out):
            model.show_result(
                image,
                result,
                out_file=dist_path / image_path.name,
                score_thr=score_thr,
                text_color=TEXT_COLOR,
                bbox_color=BBOX_COLOR,
            )

        if (overlap_det):
            # Returns non-overlapping org img, and clears masks of overlapping bubbles
            non_overlap_org_img = overlapDetection(masks, bboxes, image, ref_background_arr, model)

        # #Sums total area of bubbles in image.
        # mask_areas = np.sum(masks, axis=(1, 2))       
        # # Prints the calculated min and max area of bubbles
        # print(
        #     f"{image_path} {len(masks)} bubbles detected"
        #     f"(max area: {np.max(mask_areas)}[px^2], min area: {np.min(mask_areas)}[px^2])"
        # )

        # # Removes Overlaping Idxs from masks and bboxes
        # non_overlap_indices = [i for i in range(len(masks)) if i not in overlap_idxs]
        # masks = masks[non_overlap_indices]
        # # bboxes = bboxes[non_overlap_indices]

        #Combines all masks into one np array (combined_mask)
        #Replaces all values non 0 in np array with 255 (combined_mask)
        #Creates an image from the combined_mask np array (Both combined_mask -> mask_img & image -> image)
        #Converts mask_img to greyscale image (mask_img)
        #Invert Image (mask_img)
        combined_mask = np.sum(masks, axis=0).astype(np.uint8)
        combined_mask[combined_mask > 0 ] = 255
        mask_img = Image.fromarray(combined_mask)
        image = Image.fromarray(image)
        mask_img = mask_img.convert('L')
        mask_img = ImageOps.invert(mask_img)

        #Saves mask_img to runs folder under its original name.
        # mask_img.save(f'{dist_path}/{image_path.stem}.jpg')
        mask_img.save(f'{dist_path}/{image_path.stem}_mask.jpg')
        image.save(f'{dist_path}/{image_path.stem}_org.jpg')

        if (overlap_det):
            non_overlap_org_img = Image.fromarray(non_overlap_org_img)
            non_overlap_org_img.save(f'{dist_path}/{image_path.stem}_non_overlap.jpg')

        # Prints current Image number
        print(f"Image #{k}: {image_path.stem}")
        k = k+1


def main() -> None:
    args = get_args()
    detect(
        args.overlap_det,
        args.ono_out,
        args.config_path,
        args.checkpoint_path,
        args.source_path,
        args.name,
        args.score_thr,
        args.device,
    )


if __name__ == "__main__":
    main()
