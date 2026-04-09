import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from sklearn.cluster import DBSCAN


def sample_points_from_polygon(polygon: np.ndarray, num_points: int) -> List[Tuple[int, int]]:
    if num_points <= 0:
        return []
        
    perimeter = cv2.arcLength(polygon, closed=True)
    if perimeter < 1.0:
        return []

    sampled_points = []
    distance_interval = perimeter / num_points
    current_distance = 0.0
    for i in range(len(polygon)):
        p1 = polygon[i][0]
        p2 = polygon[(i + 1) % len(polygon)][0]
        edge_vector = p2 - p1
        edge_length = np.linalg.norm(edge_vector)
        
        if edge_length == 0:
            continue
       
        while current_distance <= edge_length:
            ratio = current_distance / edge_length
            sp_x = int(p1[0] + ratio * edge_vector[0])
            sp_y = int(p1[1] + ratio * edge_vector[1])
            sampled_points.append((sp_x, sp_y))

            if len(sampled_points) == num_points:
                return sampled_points
            current_distance += distance_interval
            
        current_distance -= edge_length
        
    return sampled_points


def filter_heatmap_and_find_centroids(
    heatmap_np: np.ndarray, 
    min_area_threshold: int = 50
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
   
    if heatmap_np.dtype != np.float32:
        heatmap_np = heatmap_np.astype(np.float32)

    heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    threshold_value, binary_mask = cv2.threshold(
        heatmap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8, ltype=cv2.CV_32S
    )

    control_points = []
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
            center_x, center_y = centroids[i]
            control_points.append((int(center_x), int(center_y)))
            filtered_mask[labels == i] = 255
    
    return control_points, binary_mask, filtered_mask


def filter_heatmap_and_find_control_points(
    heatmap_np: np.ndarray, 
    min_area_threshold: int = 50,
    num_boundary_points: int = 4
) -> Tuple[List[List[Tuple[int, int]]], np.ndarray, np.ndarray]:
    
    if heatmap_np.dtype != np.float32:
        heatmap_np = heatmap_np.astype(np.float32)
    heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    threshold_value, binary_mask = cv2.threshold(
        heatmap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8, ltype=cv2.CV_32S
    )

    control_points_groups = []
    filtered_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
            
            center_x, center_y = centroids[i]
            centroid_point = (int(center_x), int(center_y))
            component_points = np.column_stack(np.where(labels == i)[::-1])
        
            if len(component_points) < 3:
                control_points_groups.append([centroid_point])
                filtered_mask[labels == i] = 255
                continue

            hull = cv2.convexHull(component_points)
            boundary_points = sample_points_from_polygon(hull, num_boundary_points)
            
            prompt_group = [centroid_point] + boundary_points
            control_points_groups.append(prompt_group)

            filtered_mask[labels == i] = 255
            
    print(f"find {len(control_points_groups)} groups of control points。")
    return control_points_groups, binary_mask, filtered_mask


def cluster_centroids_for_prompts(
    heatmap_np: np.ndarray,
    min_area_threshold: int = 50,
    distance_threshold: int = 100
) -> Tuple[List[List[Tuple[int, int]]], np.ndarray]:
    
    if heatmap_np.dtype != np.float32:
        heatmap_np = heatmap_np.astype(np.float32)
    heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(heatmap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8, ltype=cv2.CV_32S
    )

    all_valid_centroids = []
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
            all_valid_centroids.append(centroids[i])
            filtered_mask[labels == i] = 255
    
    if not all_valid_centroids:
        print("[!] do not find any centroids!")
        return [], filtered_mask

    print(f"[*] find {len(all_valid_centroids)} initial centroids")
    
    db = DBSCAN(eps=distance_threshold, min_samples=1).fit(all_valid_centroids)
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    print(f"[*] cluster {num_clusters} groups of centroids")

    grouped_points = {}
    for i, label in enumerate(cluster_labels):
        if label not in grouped_points:
            grouped_points[label] = []

        point = tuple(int(coord) for coord in all_valid_centroids[i])
        grouped_points[label].append(point)

    prompt_groups = list(grouped_points.values())
    
    return prompt_groups, filtered_mask


def get_mask_and_label_from_sam(image: np.ndarray, point: tuple[int, int]) -> tuple[np.ndarray, str]:
    
    print(f"    - [SAM Mock] processing {point}...")
    
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center=point, radius=80, color=255, thickness=-1)  
    label = f"object_at_{point[0]}_{point[1]}"
  
    return mask, label


def iterative_object_extraction(image: np.ndarray, control_points: list[tuple[int, int]]) -> list[dict]:

    unprocessed_points = list(control_points) 
    found_objects = []

    loop_count = 0
    while unprocessed_points:
        loop_count += 1
        print(f"\n--- begin {loop_count} interation ---")

        current_point = unprocessed_points.pop(0)
        print(f"[*] process: {current_point}, remain {len(unprocessed_points)} points")

        mask, label = get_mask_and_label_from_sam(image, current_point)
        found_objects.append({
            "mask": mask,
            "label": label,
            "source_point": current_point
        })

        points_to_keep = []
        for point in unprocessed_points:
            y, x = point[1], point[0]
            if mask[y, x] == 0:
                points_to_keep.append(point)
            else:
                print(f"    - cue {point} has beed filtered by label {label}")

        unprocessed_points = points_to_keep
        
    print(f"\n--- interation success! Fine total {len(found_objects)} evidence ---")
    return found_objects


def visualize_highlighted_regions(
    original_image: Image.Image,
    heatmap: np.ndarray,
    filtered_mask: np.ndarray,
    control_points: list[tuple[int, int]]
):
   
    image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax = axes[0]
    im = ax.imshow(heatmap, cmap='hot')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.imshow(filtered_mask, cmap='gray')
    ax.axis('off')

    ax = axes[2]
    overlay = image_bgr.copy()
    highlight_color = [0, 0, 255] # BGR for Red
    overlay[filtered_mask == 255] = highlight_color
    
    final_image = cv2.addWeighted(overlay, 0.5, image_bgr, 0.5, 0)

    if control_points:
        for x, y in control_points:
            cv2.drawMarker(final_image, (x, y), (0, 255, 0), 
                           markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    
    ax.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()