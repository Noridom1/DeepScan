import base64
import json
import io
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import cv2


def is_enabled(sample) -> bool:
    """Check if artifact saving is enabled for this sample."""
    return hasattr(sample.args, "artifact_run_dir") and sample.args.artifact_run_dir is not None


def stage_dir(sample, stage_name: str) -> Path:
    """Get the directory for a specific stage, creating it if needed."""
    if not is_enabled(sample):
        return None
    path = Path(sample.args.artifact_run_dir) / stage_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_base64_image(path: Path, image_b64: str) -> None:
    """Save a base64-encoded image to disk."""
    if path is None:
        return
    image_bytes = base64.b64decode(image_b64)
    with open(path, "wb") as f:
        f.write(image_bytes)


def save_pil_image(path: Path, image: Image.Image) -> None:
    """Save a PIL Image to disk."""
    if path is None:
        return
    image.save(path)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Save a JSON object to disk."""
    if path is None:
        return
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def save_numpy_array(path: Path, array: np.ndarray) -> None:
    """Save a NumPy array to disk."""
    if path is None:
        return
    np.save(path, array)


def save_heatmap_overlay(path: Path, image_b64: str, heatmap_np: np.ndarray, cmap="jet") -> None:
    """Save a colorized heatmap and overlay it on the image."""
    if path is None:
        return

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)

    if image_np.dtype == np.uint8 and image_np.max() > 1:
        image_np = image_np.astype(np.float32) / 255.0

    h, w = heatmap_np.shape[:2]
    image_resized = cv2.resize(image_np, (w, h))

    heatmap_normalized = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)

    colormap = cv2.COLORMAP_JET
    heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    alpha = 0.5
    overlay = cv2.addWeighted(
        (image_resized * 255).astype(np.uint8),
        1 - alpha,
        heatmap_colored,
        alpha,
        0
    )

    overlay_image = Image.fromarray(overlay)
    overlay_image.save(path)


def save_mask_overlay(path: Path, image_b64: str, mask_np: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> None:
    """Save a mask overlaid on the image."""
    if path is None:
        return

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)

    if image_np.dtype == np.uint8 and image_np.max() > 1:
        image_np = image_np.astype(np.float32) / 255.0

    h, w = image_np.shape[:2]

    if mask_np.dtype != np.uint8:
        mask_np = (mask_np > 0).astype(np.uint8) * 255

    if mask_np.shape[:2] != (h, w):
        mask_np = cv2.resize(mask_np, (w, h))

    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask_np > 0] = color

    image_uint8 = (image_np * 255).astype(np.uint8) if image_np.max() <= 1 else image_np.astype(np.uint8)

    overlay = cv2.addWeighted(image_uint8, 1 - alpha, mask_rgb, alpha, 0)

    overlay_image = Image.fromarray(overlay)
    overlay_image.save(path)


def save_components_overlay(
    path: Path,
    image_b64: str,
    component_mask: np.ndarray,
    points: Optional[List[Tuple[int, int]]] = None,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None
) -> None:
    """Save connected components with optional points and boxes overlaid on the image."""
    if path is None:
        return

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)

    if image_np.dtype == np.uint8 and image_np.max() > 1:
        image_np = image_np.astype(np.float32) / 255.0

    h, w = image_np.shape[:2]

    if component_mask.dtype != np.uint8:
        component_mask = (component_mask > 0).astype(np.uint8) * 255

    if component_mask.shape[:2] != (h, w):
        component_mask = cv2.resize(component_mask, (w, h))

    image_uint8 = (image_np * 255).astype(np.uint8) if image_np.max() <= 1 else image_np.astype(np.uint8)

    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[component_mask > 0] = (0, 255, 0)

    overlay = cv2.addWeighted(image_uint8, 0.7, mask_rgb, 0.3, 0)
    overlay_cv = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    if points:
        for point in points:
            x, y = point
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(overlay_cv, (int(x), int(y)), 5, (0, 0, 255), -1)

    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay_cv, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    overlay_cv = cv2.cvtColor(overlay_cv, cv2.COLOR_BGR2RGB)
    overlay_image = Image.fromarray(overlay_cv)
    overlay_image.save(path)


def save_bbox_overlay(
    path: Path,
    image_b64: str,
    boxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None
) -> None:
    """Save bounding boxes overlaid on the image."""
    if path is None:
        return

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    h, w = image_cv.shape[:2]

    colors = [
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 0, 255),    # red
        (255, 255, 0),  # cyan
        (255, 0, 255),  # magenta
        (0, 255, 255),  # yellow
    ]

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        color = colors[idx % len(colors)]
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)

        if labels and idx < len(labels):
            label = labels[idx]
            cv2.putText(image_cv, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(image_cv)
    result_image.save(path)


def save_text_file(path: Path, text: str) -> None:
    """Save text content to a file."""
    if path is None:
        return
    with open(path, "w") as f:
        f.write(text)


def save_search_tree(path: Path, all_nodes: list) -> None:
    """Serialise the MCTS tree as a flat node list (no image data).

    Node IDs match the indices used in reasoning/node_XX/ artifact dirs,
    because both use the same all_nodes list from get_final_answer().
    Parent lookup uses object identity via id(node).
    """
    if path is None:
        return

    id_map = {id(node): idx for idx, node in enumerate(all_nodes)}

    nodes_out = []
    for idx, node in enumerate(all_nodes):
        parent_id = None
        action_taken = None
        if node.parent is not None:
            parent_id = id_map.get(id(node.parent))
            for act, child in node.parent.children.items():
                if child is node:
                    action_taken = act
                    break

        nodes_out.append({
            "id": idx,
            "parent_id": parent_id,
            "action_taken": action_taken,
            "depth": node.state["depth"],
            "region_coords": node.state.get("region_coords"),
            "valid_area_ratio": node.valid_area_ratio,
            "leaf_reward": node.leaf_reward,
            "visits": node.visits,
            "value": node.value,
            "action_history": node.state.get("action_history", []),
            "confirmed_objects": node.state.get("caption", ""),
            "missing_objects": node.state.get("missing_objects", []),
        })

    with open(path, "w") as f:
        json.dump({"nodes": nodes_out}, f, indent=2, default=str)
