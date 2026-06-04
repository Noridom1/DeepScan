import base64, math, io, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Dict, Any
from .client import get_heatmap, get_mask_point
from .control_point_sam import filter_heatmap_and_find_centroids, visualize_highlighted_regions, filter_heatmap_and_find_control_points, cluster_centroids_for_prompts
from log import get_logger

logger = get_logger("grounding")


def filter_points_by_mask(
    points_to_filter: List[Tuple[int, int]], 
    existing_mask: np.ndarray
) -> List[Tuple[int, int]]:
   
    kept_points = []
    if existing_mask is None or existing_mask.size == 0:
        return points_to_filter

    for point in points_to_filter:
        x, y = point
        h, w = existing_mask.shape[:2]
        if 0 <= y < h and 0 <= x < w and existing_mask[y, x] == 0:
            kept_points.append(point)
            
    return kept_points


def filter_groups_by_mask(
    groups_to_filter: List[List[Tuple[int, int]]],
    existing_mask: np.ndarray
) -> List[List[Tuple[int, int]]]:
    
    kept_groups = []
    if existing_mask is None or existing_mask.size == 0:
        return groups_to_filter

    for group in groups_to_filter:
        if not group:  
            continue
    
        centroid = group[0]
        x, y = centroid
        h, w = existing_mask.shape[:2]

        if 0 <= y < h and 0 <= x < w and existing_mask[y, x] == 0:
            kept_groups.append(group)

    return kept_groups


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    
    if union == 0:
        return 0.0
        
    return intersection / union


def get_bbox_from_mask_numpy(mask):
    rows, cols = np.where(mask > 0)
 
    if rows.size == 0 or cols.size == 0:
        return None
  
    x_min = np.min(cols)
    y_min = np.min(rows)
    x_max = np.max(cols)
    y_max = np.max(rows)
    
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def merge_overlapping_masks(
    found_objects: List[Dict[str, Any]], 
    iou_threshold: float = 0.75
) -> List[Dict[str, Any]]:

    if not found_objects:
        return []

    objects_to_process = found_objects.copy()
    merged_objects = []
    
    while objects_to_process:
        base_object = objects_to_process.pop(0)
        base_mask = base_object['mask']
        
        i = 0
        while i < len(objects_to_process):
            other_object = objects_to_process[i]
            other_mask = other_object['mask']
            
            iou = calculate_iou(base_mask, other_mask)
            
            if iou >= iou_threshold:
                base_mask = np.logical_or(base_mask, other_mask)
                objects_to_process.pop(i)
               
            else:
                i += 1
        
        kernel = np.ones((5, 5), np.uint8)
        final_merged_mask = cv2.morphologyEx((base_mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        base_object['mask'] = final_merged_mask
        merged_objects.append(base_object)
        
    return merged_objects


def iterative_segmentation_from_heatmap(
    initial_points: List[Tuple],
    image_b64: str,
    sam_endpoint: str = "http://127.0.0.1:8202/sam2/point_predict",
    artifact_sink=None
) -> List[Dict[str, Any]]:
   
    found_objects = []
    iteration_count = 0
    
    while initial_points:
        iteration_count += 1
        current_point = initial_points.pop(0)
    
        mask_np = get_mask_point(image_b64, current_point, endpoint=sam_endpoint)
        if mask_np is None or mask_np.size == 0 or np.all(mask_np == 0):
            logger.warning("point %s produced empty SAM mask, skipping", current_point)
            continue
            
        found_objects.append({
            "mask": mask_np,
            "source_point": current_point
        })
     
        initial_points = filter_points_by_mask(initial_points, mask_np)

    merged_objects = merge_overlapping_masks(found_objects, iou_threshold=0.3)
    for obj in merged_objects:
        mask = obj["mask"]
        ys, xs = np.where(mask > 0)
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        bbox_mask = np.zeros_like(mask, dtype=np.uint8)
        bbox_mask[y1:y2 + 1, x1:x2 + 1] = 255
        obj["mask"] = bbox_mask
        obj["area"] = np.count_nonzero(bbox_mask)

    merged_objects.sort(key=lambda o: o["area"])
    for obj in merged_objects:
        obj.pop("area", None)

    return merged_objects


def grounding(img: str,
              question: str,
              BLOCK: int,
              artifact_sink=None):

    # resized_img, heatmap = get_heatmap(img, question, endpoint = "http://localhost:8100/attention_map", block=BLOCK)
    resized_img, heatmap = get_heatmap(
        img,
        question,
        endpoint="http://localhost:8102/attention_map",
        block=BLOCK,
    )
    points, raw_binary, final_mask = filter_heatmap_and_find_centroids(heatmap)
    found_objects = iterative_segmentation_from_heatmap(points, resized_img, artifact_sink=artifact_sink)

    pil_image = Image.open(io.BytesIO(base64.b64decode(resized_img)))
    resized_width, resized_height = pil_image.size

    # Export hierarchical scanning artifacts
    if artifact_sink:
        from . import artifacts
        stage_dir = artifacts.stage_dir(artifact_sink, "hierarchical-scanning")

        # Save original and resized images
        artifacts.save_base64_image(stage_dir / "00_original.png", img)
        artifacts.save_base64_image(stage_dir / "01_resized_scan_image.png", resized_img)

        # Save heatmap
        artifacts.save_numpy_array(stage_dir / "02_attention_heatmap.npy", heatmap)

        # Colorize and save heatmap
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_image = Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        artifacts.save_pil_image(stage_dir / "02_attention_heatmap.png", heatmap_image)

        # Save attention overlay
        artifacts.save_heatmap_overlay(stage_dir / "03_attention_overlay.png", resized_img, heatmap)

        # Save raw binary mask
        raw_binary_image = Image.fromarray((raw_binary > 0).astype(np.uint8) * 255)
        artifacts.save_pil_image(stage_dir / "04_raw_binary_mask.png", raw_binary_image)

        # Save connected components
        artifacts.save_components_overlay(
            stage_dir / "05_connected_components.png",
            resized_img,
            final_mask,
            points=points
        )

        # Save anchor points
        artifacts.save_components_overlay(
            stage_dir / "06_anchor_points.png",
            resized_img,
            final_mask,
            points=points
        )

    for obj_idx, obj in enumerate(found_objects):
        mask = obj["mask"]
        bbox = get_bbox_from_mask_numpy(mask)
        x_min, y_min, x_max, y_max = bbox
        if x_min >= x_max or y_min >= y_max:
            continue

        obj['bbox'] = bbox
        cropped_img = pil_image.crop((x_min, y_min, x_max + 1, y_max + 1))
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="PNG")
        cropped_img_base64 = base64.b64encode(buffered.getvalue()).decode()
        obj['crop_img'] = cropped_img_base64

        # Export proposal artifacts
        if artifact_sink:
            from . import artifacts
            stage_dir = artifacts.stage_dir(artifact_sink, "hierarchical-scanning/proposals")

            # Save mask
            mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
            artifacts.save_pil_image(stage_dir / f"proposal_{obj_idx:02d}_mask.png", mask_image)

            # Save overlay
            artifacts.save_mask_overlay(
                stage_dir / f"proposal_{obj_idx:02d}_overlay.png",
                resized_img,
                mask
            )

            # Save crop
            artifacts.save_base64_image(stage_dir / f"proposal_{obj_idx:02d}_crop.png", cropped_img_base64)

            # Save metadata
            source_point = obj.get('source_point', None)
            artifacts.save_json(stage_dir / f"proposal_{obj_idx:02d}.json", {
                "proposal_index": obj_idx,
                "bbox_resized_frame": bbox,
                "source_point": source_point,
                "mask_shape": mask.shape,
                "crop_size": (cropped_img.width, cropped_img.height)
            })

    return resized_img, resized_width, resized_height, found_objects