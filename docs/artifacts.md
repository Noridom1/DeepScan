# DeepScan Artifact Reference

This document describes every file written under `artifacts/<sample_id>/` when
`--save-artifacts` is enabled.  It explains what each file represents, how it
maps to the three pipeline stages described in the paper, and the coordinate
conventions used throughout.

---

## How to generate artifacts

```bash
python code/src/run_single_sample.py \
  --question-file playground/data/eval/vstar/test_questions.tsv \
  --index 67 \
  --method_name vstar.oursmcts \
  --model-path <ckpt> \
  --image-size 5000 \
  --save-artifacts true \
  --artifact-dir artifacts
```

Each run writes to `artifacts/<sample_id>/`.  With `--all-rounds` the subdirectory
becomes `artifacts/<sample_id>/round-<N>/`.  Artifacts are **never** written
during batch evaluation (`run.py`) unless you add the same flags there; by
default batch runs are unaffected.

---

## Top-level structure

```
artifacts/
  <sample_id>/
    hierarchical-scanning/   Stage 1 – bottom-up region discovery
    refocusing/              Stage 2 – MCTS-guided view search
    reasoning/               Stage 3 – evidence-enhanced final answering
```

---

## Stage 1 · Hierarchical Scanning

### Paper concept

The paper's Hierarchical Scanning stage replaces the naive "look at the whole
image" approach with a coarse-to-fine search.  A BLIP-based attention model
highlights which spatial regions are most relevant to the question.  Those
regions are thresholded into binary masks, connected components are extracted,
and each component's centroid seeds a SAM2 segmentation.  The resulting
segments are proposed as candidate evidence regions.  An LVLM binary judge
then filters them, keeping only those that visibly contain clues for answering
the question.  The accepted proposals are merged into a single union bounding
box that seeds Stage 2.

### Code entry points

| Step | Function | File |
|---|---|---|
| BLIP attention map | `get_heatmap()` | `client.py` |
| Threshold + centroids | `filter_heatmap_and_find_centroids()` | `control_point_sam.py` |
| SAM2 segmentation | `iterative_segmentation_from_heatmap()` | `visual_grounding.py` |
| Full pipeline | `grounding()` | `visual_grounding.py` |
| Binary judgement | `OursMCTSQuestionSample.justify()` | `ours_mcts.py` |

### Files

#### Scanning pipeline (numbered 00–08)

| File | What it is |
|---|---|
| `00_original.png` | The raw image from the dataset row, decoded from base64.  This is the ground truth coordinate space. |
| `01_resized_scan_image.png` | The image after BLIP's internal resize (controlled by `BLOCK=640` or `BLOCK=768` depending on category).  All heatmap and proposal coordinates are in **this frame**. |
| `02_attention_heatmap.npy` | Raw float32 NumPy array of BLIP Grad-CAM attention weights.  Shape matches `01_resized_scan_image.png`.  Load with `np.load(path)`. |
| `02_attention_heatmap.png` | Colorized (JET colormap) version of the heatmap for visual inspection. |
| `03_attention_overlay.png` | Heatmap blended at α=0.5 over `01_resized_scan_image.png`.  Shows which spatial regions the BLIP model associated with the question. |
| `04_raw_binary_mask.png` | Output of Otsu thresholding on the heatmap.  White pixels are above threshold.  This is the input to connected-component analysis. |
| `05_connected_components.png` | Binary mask of the filtered connected components overlaid on `01_resized_scan_image.png` (green), with centroid points (red dots).  Each surviving component corresponds to one anchor point for SAM2. |
| `06_anchor_points.png` | Same as `05` — centroid points overlaid on the component mask.  Kept separate for comparison. |
| `07_confirmed_union_original.png` | Original image (`00_original.png`) with accepted union bbox drawn on it. |
| `07_confirmed_union_original_bbox.png` | Same as above with bounding box only — no full image overlay. |
| `08_initial_refocus_view.png` | The crop of `00_original.png` using the accepted union bbox.  This is exactly what is passed to Stage 2 as the root state.  If no proposals were accepted, this equals `00_original.png`. |

#### Scalar metadata

**`confirmed_union_resized.json`**
```json
{
  "flag": true,              // true = at least one proposal was accepted
  "union_bbox_resized": [...], // [x1, y1, x2, y2] in 01_resized_scan_image frame
  "confirmed_count": 6         // number of accepted proposals
}
```

**`confirmed_union_original.json`**
```json
{
  "flag": true,
  "union_bbox_original": [0, 0, 2249, 1012],  // [x1,y1,x2,y2] in 00_original frame
  "original_image_size": [2250, 1500]
}
```
When `flag=false` a `"message"` field replaces `union_bbox_original`.

**`binary_judgement_prompt.txt`** — the exact prompt sent to the LVLM for all
proposal judgements.  Varies by model family (Qwen3-VL uses a different
phrasing than other models).

#### Proposals subfolder

Each SAM2 segment generates four files named `proposal_NN_*` plus one JSON.

| File | Content |
|---|---|
| `proposal_NN.json` | Geometry metadata (see schema below) |
| `proposal_NN_mask.png` | Binary mask after SAM2 + morphological dilation + bbox conversion.  White = masked region.  **In `01_resized_scan_image` frame.** |
| `proposal_NN_overlay.png` | Mask blended over `01_resized_scan_image.png`. |
| `proposal_NN_crop.png` | Tight crop of `01_resized_scan_image.png` using the mask's bounding box.  This is the image shown to the binary LVLM judge. |
| `proposal_NN_judgement.json` | Judge verdict (see schema below) |

**`proposal_NN.json` schema**
```json
{
  "proposal_index": 5,
  "bbox_resized_frame": [753, 1079, 927, 1370],  // [x1,y1,x2,y2] in 01_resized frame
  "source_point": [839, 1133],                   // SAM2 prompt point (x, y) in resized frame
  "mask_shape": [1920, 2560],                    // (H, W) of the full mask array
  "crop_size": [175, 292]                        // (W, H) of proposal_NN_crop.png
}
```

**`proposal_NN_judgement.json` schema**
```json
{
  "proposal_index": 0,
  "bbox_resized_frame": [2452, 1218, 2559, 1358],
  "accepted": false,          // true if "yes" appears in the LVLM response
  "raw_response": "...",      // full LVLM text
  "prompt": "..."             // same as binary_judgement_prompt.txt
}
```

---

## Stage 2 · Refocusing

### Paper concept

Given the initial evidence crop V_e from Stage 1, the Refocusing stage searches
for V\*, the **smallest crop that still contains all objects needed to answer
the question**.  The objective is:

```
reward(node) = 1 - valid_area_ratio   if all key objects are confirmed present
             = 0                       otherwise
```

A smaller sufficient crop gets a higher reward, pushing the search toward tight
but complete views.

### Code implementation: MCTS

The paper describes this as a best-first search; the implementation uses
**Monte Carlo Tree Search (MCTS)** with `n_simulations=6` and `max_depth=3`.

Each MCTS node represents one **image crop** (a rectangular region of the
original image).  Two actions expand a node:

| Action | What it does |
|---|---|
| `repeat_question` | Asks the visual expert (LangSAM/GroundingDINO) "where are the key objects?" in the current crop.  Zooms into the union of detected bounding boxes (+ 32-pixel padding). |
| `zoom_out` | Expands the current crop by 1.5× from its center.  If missing objects were identified in the previous simulation, the expert is queried for those objects specifically and the region is unioned with any detections. |

**Simulation**: For each leaf node, the LVLM is asked "Is there a `<object>` in
this image?" for each key object.  A node passes simulation only if all key
objects are confirmed present.

**Backpropagation**: Reward propagates up to the root via standard MCTS update
(`visits += 1`, `value += reward`).

The **MCTS root** is not the initial crop directly — it is the result of the
first `repeat_question` call on the initial crop.  This means the first action
executor call (step_000) always produces the root node.

### Files

#### Initial view

**`00_initial_view.png`** — the crop passed to MCTS as the starting state.
Equals `hierarchical-scanning/08_initial_refocus_view.png`.  When Stage 1
found no accepted proposals (`flag=false`), this is the full original image.

**`00_initial_view.json`**
```json
{
  "depth": 0,
  "region_coords": [238, 519, 2249, 1499],  // [x1,y1,x2,y2] in original frame
  "image_width": 2250,
  "image_height": 1500
}
```

#### Step artifacts

Every call to an action executor writes one `step_NNN_<action>/` directory.
Steps are numbered sequentially in execution order across all simulations.
Step 000 is always the first `repeat_question` that creates the MCTS root.

##### `step_NNN_repeat_question/`

| File | Content |
|---|---|
| `parent.png` | The crop the action was applied to (parent node's image) |
| `expert_overlay.png` | `parent.png` with the visual expert's detection boxes drawn on it.  Only written when boxes were found. |
| `child.png` | The resulting child crop after zooming into the expert boxes. |
| `step.json` | Full metadata (see schema below) |

**`step.json` for `repeat_question`**
```json
{
  "step": 0,
  "action": "repeat_question",
  "parent_depth": 0,
  "parent_region_coords": [0, 0, 2249, 1012],
  "expert_boxes_parent_frame": [[1055, 737, 1148, 825], [338, 762, 553, 866]],
  "expert_boxes_found": true,
  "child_region_coords": [306, 705, 1180, 898],
  "valid_area_ratio": 0.050,
  "action_history": ["Repeat the question."]
}
```

- `parent_region_coords`: the parent node's location in the **original image** frame.
- `expert_boxes_parent_frame`: GroundingDINO detections in the **parent's local pixel frame** (origin at top-left of `parent.png`, not the original image).
- `child_region_coords`: the resulting child region in the **original image** frame.

##### `step_NNN_zoom_out/`

| File | Content |
|---|---|
| `parent.png` | The parent node's crop before any expansion |
| `zoom_candidate.png` | The 1.5× expanded crop, before expert-guided correction |
| `expert_overlay.png` | `zoom_candidate.png` with expert detections drawn on it.  Only written when boxes were found. |
| `child.png` | The final child crop.  Equals `zoom_candidate.png` when no region correction occurred. |
| `step.json` | Full metadata (see schema below) |

**`step.json` for `zoom_out`**
```json
{
  "step": 2,
  "action": "zoom_out",
  "parent_depth": 1,
  "parent_region_coords": [306, 705, 1180, 898],
  "zoom_candidate_region_coords": [87, 656, 1399, 946],
  "expert_mode": "key_objects",
  "expert_query": "scarf",
  "expert_boxes_zoom_candidate_frame": [[235, 44, 587, 289]],
  "expert_boxes_found": true,
  "child_region_coords": [87, 656, 1399, 946],
  "valid_area_ratio": 0.1127,
  "action_history": ["Repeat the question.", "Zoom out the region by 1.5x"]
}
```

- `zoom_candidate_region_coords`: the 1.5× expanded area in the **original image** frame.
- `expert_mode`: `"missing_objects"` when the parent's simulation failed and missing objects are known; `"key_objects"` otherwise.
- `expert_boxes_zoom_candidate_frame`: detections in the **zoom candidate's local pixel frame** (origin at top-left of `zoom_candidate.png`).
- When `expert_mode="missing_objects"` and boxes were found, `child_region_coords` is the union of the parent region and the expert detections — it may differ from `zoom_candidate_region_coords`.  When `expert_mode="key_objects"`, the region is not changed and `child_region_coords == zoom_candidate_region_coords`.

#### Search tree

**`search_tree.json`** — written after all simulations complete, once every
node's `leaf_reward`, `visits`, and `value` are finalised.

Format: a flat list of node entries.  Node IDs match the folder numbers in
`reasoning/node_XX/` because both use the same traversal order.

```json
{
  "nodes": [
    {
      "id": 0,
      "parent_id": null,         // null for root
      "action_taken": null,      // action key that produced this node from its parent
      "depth": 1,
      "region_coords": [306, 705, 1180, 898],   // original image frame
      "valid_area_ratio": 0.050, // fraction of original image area this crop covers
      "leaf_reward": 0,          // 1 - valid_area_ratio if all objects confirmed, else 0
      "visits": 5,               // MCTS visit count
      "value": 1.664,            // cumulative reward from backpropagation
      "action_history": ["Repeat the question."],
      "confirmed_objects": "scarf",  // objects verified present during simulation
      "missing_objects": []          // objects not found during simulation
    },
    {
      "id": 1,
      "parent_id": 0,
      "action_taken": "zoom_out",
      ...
    }
  ]
}
```

**Reading the tree:** `parent_id` and `action_taken` together reconstruct the
edge.  A node with `parent_id=null` is the MCTS root (depth 1, not the initial
crop which has depth 0).  The best node is the one with the highest
`leaf_reward`; it corresponds to V\* in the paper.

**`valid_area_ratio` and reward intuition:**
A node covering 1% of the original image that still contains all key objects
gets `reward = 0.99`.  A node covering 50% that confirms objects gets
`reward = 0.50`.  This drives the search toward tighter, more precise crops.
A node that fails simulation (missing objects) always gets `reward = 0` regardless
of size — size is only rewarded when completeness is satisfied.

---

## Stage 3 · Evidence-Enhanced Reasoning

### Paper concept

Rather than asking the LVLM to answer from the original image alone, DeepScan
assembles a **multi-image evidence bundle** combining the best crop V\* with
earlier evidence crops and the full original image.  This gives the LVLM both
a zoomed-in view of the relevant region and the broader scene context.

The code extends this by running inference on **all nodes** explored during the
MCTS search (not just V\*) and collecting weighted votes, where each node's
vote is weighted by its `leaf_reward`.  The option with the highest total weight
is the final answer.

### Image bundle composition (per model family)

| Model | Condition | Images sent to LVLM |
|---|---|---|
| Qwen3-VL | always | `[node_image]` only |
| Other | `flag=True`, node is initial crop | `[node_image, original_image]` |
| Other | `flag=True`, deeper node | `[node_image, initial_evidence_image, original_image]` |
| Other | `flag=False`, node is original | `[original_image]` |
| Other | `flag=False`, deeper node | `[node_image, original_image]` |

Qwen3-VL uses a single image per node because its visual encoder handles
scale internally; the multi-scale bundle is used for other models only.

### Files

#### Per-node subfolder: `reasoning/node_NN/`

One directory per MCTS node.  Node indices match `search_tree.json` node IDs.

| File | Content |
|---|---|
| `input_00.png` | First image in the LVLM prompt (the node's crop) |
| `input_01.png` | Second image, if any (initial evidence or original) |
| `input_02.png` | Third image, if any (original) |
| `prompt.txt` | The full text prompt (question + options + instruction) |
| `response.txt` | Raw LVLM response text |
| `metadata.json` | Structured metadata (see schema below) |

**`metadata.json` schema**
```json
{
  "node_index": 1,
  "depth": 2,
  "region_coords": [87, 656, 1399, 946],   // this node's crop in original image frame
  "leaf_reward": 0.887,
  "valid_area_ratio": 0.112,
  "action_history": ["Repeat the question.", "Zoom out the region by 1.5x"],
  "model_name": "Qwen3-VL-8B-Instruct",
  "num_reasoning_images": 1,
  "raw_response": "..."
}
```

The `action_history` shows the path from the MCTS root to this node.  Combined
with `region_coords` this uniquely locates the node in the search tree and on
the original image.

#### Summary files

**`final_prompt.txt`** — the text prompt shared by all nodes (question +
options + instruction suffix).  Identical across `node_NN/prompt.txt` files.

**`weighted_votes.json`**
```json
{
  "votes": {"A": 0.0, "B": 1.664},
  "all_answers": [["A", 0], ["B", 0.887], ["A", 0], ["B", 0.777], ["A", 0], ["A", 0]]
}
```
Each entry in `all_answers` is `[predicted_option, leaf_reward_weight]`.  The
option with the highest sum of weights is the final answer.  When all weights
are zero (every node's simulation failed), a fallback inference on the full
original image is used.

**`final_answer.json`**
```json
{
  "final_answer": "B",
  "best_node_index": 1,      // node with highest leaf_reward (= V* from paper)
  "best_node_reward": 0.887,
  "best_node_region": [87, 656, 1399, 946],
  "total_nodes_explored": 6,
  "use_ensemble": true
}
```

**`best_node.png`** — the crop image of the node with the highest `leaf_reward`
(V\* in the paper).  This is the "optimal refocused view" the search converged on.

---

## Coordinate systems

All bounding boxes are `[x1, y1, x2, y2]` (left, top, right, bottom).

| Frame | Used in | Note |
|---|---|---|
| **Original image** | `confirmed_union_original.json`, `refocusing/` step `*_region_coords`, `search_tree.json region_coords`, `reasoning/metadata.json region_coords` | Pixel space of `00_original.png` |
| **Resized scan image** | `confirmed_union_resized.json`, `proposal_NN.json bbox_resized_frame`, `proposal_NN_judgement.json bbox_resized_frame` | Pixel space of `01_resized_scan_image.png` |
| **Parent-local** | `step_NNN_repeat_question/step.json expert_boxes_parent_frame` | Origin at top-left of `parent.png`; (0,0) = top-left of the parent crop, not the original image |
| **Zoom-candidate-local** | `step_NNN_zoom_out/step.json expert_boxes_zoom_candidate_frame` | Origin at top-left of `zoom_candidate.png` |

Expert detection boxes are always in the **local frame of the image sent to the
expert** (the cropped PNG), not the original image.  To map them back to the
original image, add the crop's origin: `(box_local_x + crop_x1, box_local_y + crop_y1)`.

---

## Worked example: sample 67

**Question:** *What is the color of the scarf?*

### Stage 1 summary

- BLIP scanned the resized image (2560×1920) and produced 27 attention-weighted
  centroids.
- SAM2 segmented each centroid; 6 proposals survived.
- The binary judge accepted 6 proposals (all containing clues about the scarf).
- Union bbox in original frame: `[0, 0, 2249, 1012]` (top-half of the image).
- `08_initial_refocus_view.png` = top-half crop fed to Stage 2.

### Stage 2 summary (MCTS tree, 6 nodes)

```
node 0  depth=1  repeat_question          reward=0      (scarf not confirmed in tight crop)
  ├─ node 1  depth=2  zoom_out            reward=0.887  ← V* (best node)
  │    ├─ node 2  depth=3  repeat_question  reward=0    (scarf not found)
  │    └─ node 3  depth=3  zoom_out         reward=0.777
  └─ node 4  depth=2  repeat_question     reward=0      (scarf not found)
       └─ node 5  depth=3  zoom_out         reward=0
```

- Step 000: `repeat_question` on initial crop → node 0.  Expert found two
  regions (two people).  Zoomed to their union.  Simulation: scarf not
  confirmed → reward=0.
- Step 002: `zoom_out` on node 0 → node 1.  1.5× expansion found scarf on
  first key-objects query.  Simulation: scarf confirmed → reward=0.887.
- Steps 001, 004: further `repeat_question` zooms → tighter crops that lost
  the scarf (reward=0).
- Steps 003, 005: `zoom_out` attempts to recover → node 3 confirmed the scarf
  at a larger area (reward=0.777), node 5 still missed it.

**V\***: node 1 (`zoom_out` at depth 2, `valid_area_ratio=0.112`, region
`[87, 656, 1399, 946]` in original frame).

### Stage 3 summary

Ensemble weighted votes:
- Nodes with reward=0 voted "A" but contributed 0 weight.
- Node 1 voted "B" with weight 0.887; node 3 voted "B" with weight 0.777.
- Total: B = 1.664.  **Final answer: B**.
