import argparse
from itertools import chain
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector
from PIL import Image, ImageOps, ImageDraw
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

def removeOverlap(masks, bboxes, image, overlap_bubbles_keep):
    non_overlap_org_img = np.copy(image)

    for idx, mask in enumerate(masks):
        if idx not in overlap_bubbles_keep:
            binary_mask = masks[idx].astype(np.uint8)

            # Kernel used to "blur" or "expand" the range of the bubble to cover the edges.
            kernel = np.ones((5, 5), np.uint8)

            # Dilated Mask for where the bubble is.
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            # Replace detected overlaps with the respective area in the background.
            non_overlap_org_img[dilated_mask > 0] = ref_background_arr[dilated_mask > 0]

    # Remove Overlapping Indices from masks and bboxes
    non_overlap_indices = [i for i in range(len(masks)) if i in overlap_bubbles_keep]
    masks = masks[non_overlap_indices]

    return non_overlap_org_img, masks

# Method for deciding which bubble to keep out of a group of overlapping bubbles
# List bubbles in order of largest to smallest in arr
# Remove all but x(largest) from image
# Detect x's bounding box for a bubble, if one found, return x idx
# Otherwise, set x to next largest and repeat.
# If none are detected, simly return the largest.
def keepBubble(group, masks, bboxes, image, model, score_thr):
    bubbleSizes = []
    for idx in group:
        bubbleSizes.append((idx, getArea(bboxes[idx])))

    # Sort the bubbles by area in descending order
    bubbleSizes.sort(key=lambda x: x[1], reverse=True)

    # Iterate through the sorted bubbles for further processing
    for idx, area in bubbleSizes:
        # Get the bounding box corresponding to the current index
        bbox = bboxes[idx]

        non_overlap_org_img, masks  = removeOverlap(masks, bboxes, image, overlap_bubbles_keep=[idx])
        
        # Crop the image using the bounding box
        x_min, y_min, x_max, y_max = map(int, bbox[:4])
        # cropped_image = non_overlap_org_img[y_min:y_max, x_min:x_max]

        # Run detection of bubble's bbox portion of the image
        det_result = inference_detector(model, non_overlap_org_img)
        # det_result = inference_detector(model, cropped_image)
        det_bboxes, det_masks = det_result
        det_bboxes = det_bboxes[0]
        det_bboxes = [bbox for bbox in det_bboxes if bbox[4] > score_thr]


        # Check if any bubble was detected
        if (len(det_bboxes) > 0): 
            for det_bbox in det_bboxes:
                # If bbox doesnt overlap, go to next in det_bboxes
                # Result is that if bbox overlaps with original bubble we consider it "the same bubble" and that it was detected still
                if (isOverlapping(bbox, det_bbox)):
                    print(f"Detected: {idx}")

                    output_path = Path("/home/iec/Documents/bubble_project/BubbleProject/bubble_detector/runs/test") / f"DetBubble{idx}.jpg"

                    cropped_image_pil = Image.fromarray(cv2.cvtColor(non_overlap_org_img, cv2.COLOR_BGR2RGB))

                    # Draw the bounding boxes on the image
                    draw = ImageDraw.Draw(cropped_image_pil)
                    for det_bbox in det_bboxes:
                        x1, y1, x2, y2, score = det_bbox
                        draw.rectangle([x1, y1, x2, y2], outline="white", width=3)

                    # Save the image with bounding boxes
                    cropped_image_pil.save(output_path)

                    return idx

    print("##################################################################################")
    print(f"Not Detected: {bubbleSizes[0][0]}")
    print(bubbleSizes)
    return bubbleSizes[0][0]

def overlapDetection(masks, bboxes, image, model, score_thr):
    # Builds and creates a graph of all overlapping bubble groups
    bubble_graph = build_graph(bboxes)
    overlap_bubbles = find_connected_components(bubble_graph)

    # Decides which bubbles from each group to keep based on which is largest(by bbox)
    overlap_bubbles_keep = set()
    for group in overlap_bubbles:
        overlap_bubbles_keep.add(keepBubble(group, masks, bboxes, image, model, score_thr))

    # Use helper method to remove overlapping bubbles
    non_overlap_org_img, masks = removeOverlap(masks, bboxes, image, overlap_bubbles_keep)

    return non_overlap_org_img, masks

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
    global ref_background_arr
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
            non_overlap_org_img, masks = overlapDetection(masks, bboxes, image, model, score_thr)

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
        mask_img = mask_img.convert('L')
        mask_img = ImageOps.invert(mask_img)

        #Saves mask_img to runs folder under its original name.
        # mask_img.save(f'{dist_path}/{image_path.stem}.jpg')
        mask_img.save(f'{dist_path}/{image_path.stem}_BW.jpg')

        if (overlap_det):
            non_overlap_org_img = Image.fromarray(non_overlap_org_img)
            non_overlap_org_img.save(f'{dist_path}/{image_path.stem}_non_overlap.jpg')
            image = Image.fromarray(image)
            image.save(f'{dist_path}/{image_path.stem}_org.jpg')



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
