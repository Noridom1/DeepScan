import torch, argparse
from transformers import BertTokenizer
from lavis.models.blip_models.blip import BlipBase
import io, base64, os, torch, uvicorn, cv2, math
import numpy as np
from PIL import Image
from torchvision import transforms
from lavis.common.gradcam import getAttMap
from fastapi import FastAPI
from pydantic import BaseModel
from lavis.models import load_model, load_preprocess, load_model_and_preprocess
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam


LOCAL_TOKENIZER_PATH = "/home/phucnlt2/DeepScan/models/bert-base-uncased/" # <--- repalce with your local path
original_init_tokenizer = BlipBase.init_tokenizer
@classmethod
def patched_init_tokenizer(cls):
    print(f"--- loading local tokenizer from: {LOCAL_TOKENIZER_PATH} ---")
    tokenizer = BertTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    return tokenizer
BlipBase.init_tokenizer = patched_init_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_itm, vis_processors, text_processors = load_model_and_preprocess(
    "blip_image_text_matching", "large",
    device=device,
    is_eval=True,
)
loader = transforms.Compose([transforms.ToTensor()])


app = FastAPI(title="DyFo Attention Service")


class AttentionRequest(BaseModel):
    image: str
    question: str
    block: int

def resize_to_multiple(img: Image.Image, block: int) -> Image.Image:
    w, h = img.size
    new_w = math.ceil(w / block) * block
    new_h = math.ceil(h / block) * block
    return img.resize((new_w, new_h), Image.BICUBIC)

def genAttnMap(image, question, tensor_image, model, tokenized_text, raw_image):
    with torch.set_grad_enabled(True):
        gradcams, _ = compute_gradcam(model=model,
                            visual_input=image,
                            text_input=question,
                            tokenized_text=tokenized_text,
                            block_num=6)
    gradcams = [gradcam_[1] for gradcam_ in gradcams]
    gradcams1 = torch.stack(gradcams).reshape(image.size(0), -1)
    itc_score = model({"image": image, "text_input": question}, match_head='itc')
    ratio = 1 - itc_score/2
    ratio = min(ratio, 1-10**(-5))
    resized_img = raw_image.resize((384, 384))
    norm_img = np.float32(resized_img) / 255
    gradcam = gradcams1.reshape(24,24)
    avg_gradcam = getAttMap(norm_img, gradcam.cpu().numpy(), blur=True, overlap=False)
    return avg_gradcam
    
def compute_attention(raw_image_b64: str, question: str):
    img_bytes = base64.b64decode(raw_image_b64)
    raw_image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    tensor_image = loader(raw_image_pil.resize((384, 384)))
    image = vis_processors["eval"](raw_image_pil).unsqueeze(0).to(device)
    question_tok = text_processors["eval"](question)
    tokenized_text = model_itm.tokenizer(
        question_tok, padding="longest", truncation=True, return_tensors="pt"
    ).to(device)
    heat = genAttnMap(
        image, question_tok, tensor_image, model_itm, tokenized_text, raw_image_pil
    )
    
    buf_img = io.BytesIO()
    raw_image_pil.save(buf_img, format="PNG")
    img_b64 = base64.b64encode(buf_img.getvalue()).decode('utf-8')

    return img_b64, heat #heat_uint8


def compute_full_attention(raw_image_b64: str, question: str, block: int):
    img_bytes = base64.b64decode(raw_image_b64)
    raw_image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    resized_img = resize_to_multiple(raw_image_pil, block)
    W, H = resized_img.size
    n_cols, n_rows = W // block, H // block

    full_heatmap = np.zeros((H, W), dtype=np.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            x0, y0 = c * block, r * block
            # patch=(384,384)
            patch = resized_img.crop((x0, y0, x0 + block, y0 + block)).resize((384, 384), Image.BICUBIC)                
            tensor_image = loader(patch)
            image = vis_processors["eval"](patch).unsqueeze(0).to(device)
            q_tensor = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(
                q_tensor, padding="longest", truncation=True, return_tensors="pt"
            ).to(device)
            heat_patch = genAttnMap(
                image, q_tensor, tensor_image, model_itm, tokenized_text, patch
            )
            
            if not isinstance(heat_patch, np.ndarray):
                heat_patch = heat_patch.cpu().numpy()

            heat_patch = cv2.resize(heat_patch, (block, block), interpolation=cv2.INTER_LINEAR)
            full_heatmap[y0:y0+block, x0:x0+block] = heat_patch

    # heat_norm  = (full_heatmap - full_heatmap.min()) / (full_heatmap.ptp() + 1e-8)
    # heat_uint8 = (heat_norm * 255).astype(np.uint8) # -> np.ndarray [H,W]

    buf_resized_img = io.BytesIO()
    resized_img.save(buf_resized_img, format="PNG")
    resized_img_b64 = base64.b64encode(buf_resized_img.getvalue()).decode('utf-8')

    return resized_img_b64, full_heatmap #heat_uint8


def filter_masks_api(resized_img_b64: str, question: str, masks_b64_list: list[str]) -> list[str]:
    # Decode the base64 resized image to a PIL image and numpy array
    original_image = Image.open(io.BytesIO(base64.b64decode(resized_img_b64))).convert("RGB")
    original_np = np.array(original_image)
    
    # Compute text embedding for the question using BLIP
    text_inputs = processor(text=question, return_tensors="pt", padding=True)
    text_feat = model.get_text_features(**text_inputs)
    # (Optionally normalize the text feature for cosine similarity)
    text_vec = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
    
    # Prepare list for masked image data and embeddings
    masked_images = []
    img_embeddings = []
    
    # Decode each mask, apply to original image to get masked object image and embedding
    for mask_b64 in masks_b64_list:
        # Decode mask image (assume mask is a grayscale or RGBA image in base64)
        mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
        mask_arr = np.array(mask_img)
        # Ensure mask_arr is binary (0 or 1)
        if mask_arr.ndim == 3:
            # If mask has multiple channels (RGBA), convert to single channel
            mask_arr = mask_arr[:, :, 0]
        mask_bin = (mask_arr > 0).astype(np.uint8)
        # Apply mask to original image
        masked_np = original_np * mask_bin[:, :, None]  # broadcast mask over color channels
        masked_images.append(masked_np)
        # Get BLIP image embedding for the masked image
        img_inputs = processor(images=Image.fromarray(masked_np), return_tensors="pt")
        img_feat = model.get_image_features(**img_inputs)
        img_vec = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)  # normalize
        img_embeddings.append(img_vec)
    
    # Merge semantically similar masks (threshold > 0.9 cosine similarity)
    merged_masks = []      # to store merged mask arrays
    merged_embeds = []     # to store embedding for merged mask
    used = set()
    num_masks = len(masked_images)
    for i in range(num_masks):
        if i in used:
            continue
        # Start a new cluster with mask i
        current_mask = (masked_images[i][..., 0] > 0).astype(np.uint8)  # binary mask from first channel
        current_embed = img_embeddings[i]
        used.add(i)
        # Check others for similarity
        for j in range(i+1, num_masks):
            if j in used:
                continue
            # Compute cosine similarity between embeddings i and j
            cos_sim = float((current_embed @ img_embeddings[j].T).item())
            if cos_sim >= 0.90:
                # Merge this mask into current cluster
                mask_j = (masked_images[j][..., 0] > 0).astype(np.uint8)
                current_mask = np.maximum(current_mask, mask_j)  # union
                used.add(j)
        # Recompute embedding for the merged mask (if cluster has multiple merged)
        if np.any(current_mask != (masked_images[i][..., 0] > 0)):
            merged_img = Image.fromarray(original_np * current_mask[:, :, None])
            img_inputs = processor(images=merged_img, return_tensors="pt")
            current_embed = model.get_image_features(**img_inputs)
            current_embed = current_embed / current_embed.norm(p=2, dim=-1, keepdim=True)
        merged_masks.append(current_mask)
        merged_embeds.append(current_embed)
    
    # Filter out masks unrelated to question (cosine similarity < 0.7 with text)
    filtered_masks = []
    filtered_images_b64 = []
    for mask, img_vec in zip(merged_masks, merged_embeds):
        cos_sim_q = float((img_vec @ text_vec.T).item())
        if cos_sim_q >= 0.70:
            filtered_masks.append(mask)
            # Encode the masked image (apply mask to original and encode to base64)
            masked_img = Image.fromarray(original_np * mask[:, :, None])
            buffer = io.BytesIO()
            masked_img.save(buffer, format="PNG")
            filtered_images_b64.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    # Sort remaining masks by area (ascending)
    filtered_areas = [mask.sum() for mask in filtered_masks]
    sorted_indices = sorted(range(len(filtered_masks)), key=lambda k: filtered_areas[k])
    sorted_images_b64 = [filtered_images_b64[idx] for idx in sorted_indices]
    return sorted_images_b64


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8100, help="Service port")
args = parser.parse_args()

@app.post("/attention_map")
async def attention_map(request: AttentionRequest):
    resized_img_b64, heatmap_np = compute_full_attention(request.image, request.question, request.block)

    heatmap_bytes = heatmap_np.tobytes()
    heatmap_b64 = base64.b64encode(heatmap_bytes).decode('utf-8')
   
    return {
        "resized_img": resized_img_b64,
        "heatmap": {
            "data_b64": heatmap_b64,
            "shape": heatmap_np.shape,
            "dtype": str(heatmap_np.dtype)
        }
    }

if __name__ == "__main__":
    uvicorn.run("blip_service:app", host="0.0.0.0", port=args.port, limit_concurrency=10000, backlog=10000, log_level="debug")