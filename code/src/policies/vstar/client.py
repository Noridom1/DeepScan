import base64, io, requests, cv2
import numpy as np
from PIL import Image
from typing import List, Tuple


def get_heatmap(raw_image: str, 
                question: str,
                endpoint: str = "http://localhost:8102/attention_map",
                block: int = 786):
   
    payload = {
        "image": raw_image,
        "question": question,
        "block": block
    }
    resp = requests.post(
        endpoint,
        json=payload, 
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    resized_img_b64 = data["resized_img"]
    heatmap_info = data["heatmap"]
    decoded_bytes = base64.b64decode(heatmap_info["data_b64"])
    heatmap_np = np.frombuffer(
        decoded_bytes,
        dtype=np.dtype(heatmap_info["dtype"])
    ).reshape(heatmap_info["shape"])

    return resized_img_b64, heatmap_np


def get_mask_point(
    image_b64: str,
    positive_point: Tuple[int, int],
    endpoint: str = "http://127.0.0.1:8202/sam2/point_predict"
) -> np.ndarray:

    payload = {
        "image_b64": image_b64,
        "pos": [list(positive_point)],
        "neg": None,
        "multimask": False
    }

    try:
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=300,  
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        raise

    mask_info = data["mask"]
    decoded_bytes = base64.b64decode(mask_info["data_b64"])
    mask_np = np.frombuffer(
        decoded_bytes,
        dtype=np.dtype(mask_info["dtype"])
    ).reshape(mask_info["shape"])

    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=40)

    return mask_dilated