# DeepScan Codebase Architecture

This document maps the DeepScan paper implementation to the files in this
repository. It focuses on the method pipeline, the services used by the
implementation, the V* experiment flow, result files, and the changes needed to
run the README's two-GPU setup on physical GPUs 2 and 3.

## Paper Method to Code Map

The paper describes DeepScan as a training-free, bottom-up framework for
visually grounded reasoning in large vision-language models. In this codebase,
the method is implemented mainly under `code/src/policies/vstar/` and is exposed
through the policy name `vstar.oursmcts`.

| Paper concept | Implementation | Main files |
| --- | --- | --- |
| Hierarchical Scanning | BLIP Grad-CAM heatmap over image blocks, heatmap thresholding, point selection, SAM2 point-prompt segmentation, candidate crop extraction | `visual_grounding.py`, `client.py`, `control_point_sam.py`, `blip_service.py`, `sam2_service.py` |
| Refocusing | Question-conditioned evidence validation followed by an MCTS-style crop search with `repeat_question` and `zoom_out` actions | `ours_mcts.py`, `MCTS.py`, `model_service.py` |
| Evidence-Enhanced Reasoning | Final LVLM inference over the selected evidence image, the refocused image, and/or the original image; weighted voting over explored nodes | `MCTS.py`, `policy.py`, `qwen_runtime.py` |
| Training-free design | No fine-tuning loop is present; all modules run at inference time using frozen BLIP, LangSAM/GroundingDINO, SAM2, and Qwen-VL models | `code/scripts/*_server/`, `code/src/` |

## Runtime Entry Points

The batch evaluation entry point is `code/src/run.py`.

It reads a TSV question file, splits it with `--num-chunks` and `--chunk-idx`,
creates a `QuestionSample` from `policy_map`, runs each sample asynchronously,
and writes one JSON object per result to `--answers-file`.

The single-sample debugging entry point is `code/src/run_single_sample.py`.

It loads one row by `--index` or `--row-idx`, verifies that the TSV contains a
base64 `image` column, then runs the selected policy. This is the easiest entry
point for inspecting one V* example.

The shell entry point for V* is `code/scripts/vstar/stream_vstar_qwen.sh`.

It derives the number of chunks from `CUDA_VISIBLE_DEVICES`, starts one
`run.py` process per visible GPU, merges the shard JSONL files, and invokes the
dataset evaluation script.

## Services

DeepScan is implemented as one main LVLM process plus three local HTTP services.

### Search Expert: BLIP Grad-CAM

Files:

- `code/scripts/blip_server/blip_service.py`
- `code/scripts/blip_server/start_server.sh`
- Client calls from `code/src/policies/vstar/client.py`

Endpoint:

- `POST /attention_map`

Default README port:

- `8100`

Role in the method:

The service loads BLIP image-text matching from LAVIS and computes patch-wise
Grad-CAM maps for a question. `compute_full_attention()` resizes the image to a
multiple of the block size, scans each block, stitches patch heatmaps back into a
full-image heatmap, and returns both the resized image and the heatmap bytes.

In `visual_grounding.py`, `grounding()` calls this service, thresholds the
heatmap, extracts connected-component centroids, and uses those centroids as
SAM2 positive points.

### Visual Expert: LangSAM / GroundingDINO

Files:

- `code/scripts/expert_server/model_service.py`
- `code/scripts/expert_server/start_server.sh`
- Client calls from `code/src/policies/vstar/MCTS.py`

Endpoint:

- `POST /predict`

Default README port:

- `8000`

Role in the method:

This service performs text-conditioned detection and segmentation. The MCTS code
uses it to localize objects mentioned in the question or in missing-object
prompts. `MCTSQuestionSample.get_expert_boxes()` sends a base64 image and text
prompt to this service and expects boxes, labels, and masks.

### SAM2 Point-Prompt Service

Files:

- `code/scripts/sam2_server/sam2_service.py`
- `code/scripts/sam2_server/start_server.sh`
- Client calls from `code/src/policies/vstar/client.py` and
  `code/src/policies/vstar/visual_grounding.py`

Endpoint:

- `POST /sam2/point_predict`

Default README port:

- `8200`

Role in the method:

This service converts candidate heatmap points into binary masks. The masks are
merged, converted to bounding boxes, sorted by area, and cropped into candidate
evidence images.

## Method Flow in the Current Implementation

### 1. Sample Setup

`run.py` reads each row from the question TSV and instantiates
`vstar.oursmcts`, which maps to `OursMCTSQuestionSample` through
`code/src/policies/__init__.py` and `code/src/policies/vstar/__init__.py`.

`QuestionSample` in `policy.py` stores:

- the base64 image from `row["image"]`
- answer options from columns `A`, `B`, `C`, `D`
- the category, question, answer, and model path
- LVLM generation helpers

### 2. Key Object Extraction

`MCTSQuestionSample._process()` first calls `extract_key_objects()`.

For Qwen-style models, this uses the LVLM itself with a blank image and a text
prompt asking it to extract objects and complete descriptions from the question.
Those extracted object phrases become the object checklist used in simulation.

### 3. Hierarchical Scanning

`OursMCTSQuestionSample.get_final_answer()` performs a prefilter step before the
MCTS search:

1. Calls `grounding(self.image, self.row["question"], BLOCK=...)`.
2. Uses a larger block for relative-position questions (`768`) and a smaller
   block for other categories (`640`).
3. `grounding()` calls the BLIP search expert at `/attention_map`.
4. The heatmap is thresholded with Otsu thresholding and connected components.
5. Component centroids are sent to SAM2 as point prompts.
6. SAM2 masks are merged and converted to crop boxes.
7. Each crop is returned as base64 evidence plus a bounding box.

This implements the paper's cue discovery and local evidence proposal step.

### 4. Evidence Justification

`OursMCTSQuestionSample.justify()` asks the LVLM whether each proposed crop
contains clues needed to answer the question.

If one or more crops are accepted, their boxes are unioned and mapped from the
resized scan image back into the original image coordinate frame. That union crop
becomes the initial MCTS state. If no crop is accepted, the original image
remains the initial state.

This is the first part of refocusing: discard irrelevant proposals and start the
search from the smallest accepted evidence region available.

### 5. Refocusing with MCTS

`MCTSQuestionSample` defines the search tree:

- `max_depth = 3`
- `n_simulations = 6`
- `c_puct = 1.0`
- actions: `repeat_question`, `zoom_out`

The actions are:

- `repeat_question`: send the current image and original question to the visual
  expert, union returned boxes, pad them, and crop.
- `zoom_out`: enlarge the current crop by 1.5x. If the node has missing objects,
  ask the visual expert for those missing objects in the enlarged crop.

The simulation reward checks whether the extracted key objects are present in
the current crop. If all are present, the reward is `1 - valid_area_ratio`, so a
smaller complete view receives a higher reward. If any object is missing, reward
is `0`.

This directly corresponds to the paper's refocusing objective: keep the visual
context needed for the question while shrinking the irrelevant area.

### 6. Evidence-Enhanced Reasoning

After MCTS, `get_final_answer()` gathers all explored nodes and builds the final
multiple-choice prompt from the hint, question, and options.

For Qwen2.5-style models, the code passes multi-image evidence to the LVLM:

- accepted crop plus original image
- deeper MCTS crop plus accepted initial crop plus original image
- or just the original image when no evidence crop exists

For Qwen3-VL, the current branch sends only the node image.

Each node produces an answer letter. With `use_ensemble = True`, answers are
weighted by the node's leaf reward and the highest weighted vote becomes the
final prediction. If every vote has zero weight, the code falls back to direct
inference on the original image.

## Experiment Flow

The released experiment path is V*.

Input dataset:

- Expected under `playground/data/eval/vstar/`
- Main TSV: `test_questions.tsv`
- Required columns include `image`, `question`, `answer`, `category`, and
  multiple-choice columns such as `A`, `B`, `C`, `D`

Batch command from the README:

```bash
conda activate deepscan
bash code/scripts/vstar/stream_vstar_qwen.sh oursmcts False
```

Important script settings:

- `METHOD=${1:-clean}` means the first argument must be `oursmcts` to use
  DeepScan.
- `CKPT` must point to the local Qwen checkpoint.
- `QUESTION_FILE` must point to the local V* TSV.
- `--image-size 5000` allows large images to be passed to the local runtime.
- `--num-chunks` and `--chunk-idx` split the TSV across visible GPUs.

Output:

- Per-chunk predictions:
  `eval/vstar/deepscan/<model>/<split>/<method>/<chunks>_<idx>_<model>.jsonl`
- Merged predictions:
  `eval/vstar/deepscan/<model>/<split>/<method>/<model>.jsonl`

Each JSONL prediction contains:

- `question_id`
- `round_id`
- final multiple-choice `text`
- `options`
- `answer`
- `model_id`

The script then calls the V* `eval.py` script and prints the final evaluation
statistics. The README reports V* results of 90.6 overall for Qwen2.5-VL 7B and
92.2 overall for Qwen3-VL 8B.

## Running on Physical GPUs 2 and 3

The README examples use GPU 0 for expert services and GPU 1 for the main LVLM.
For your case, the analogous allocation is:

- GPU 2: visual expert, search expert, SAM2 service
- GPU 3: Qwen-VL main runtime

Because this code derives service ports from physical GPU IDs, using GPU 2
changes the service ports:

| Service | README GPU 0 port | GPU 2 port from current startup scripts |
| --- | ---: | ---: |
| Visual expert | `8000` | `8002` |
| BLIP search expert | `8100` | `8102` |
| SAM2 point-prompt service | `8200` | `8202` |

### Start Services on GPU 2

Run these in their respective conda environments:

```bash
CUDA_VISIBLE_DEVICES=2 bash code/scripts/expert_server/start_server.sh
CUDA_VISIBLE_DEVICES=2 bash code/scripts/blip_server/start_server.sh
CUDA_VISIBLE_DEVICES=2 bash code/scripts/sam2_server/start_server.sh
```

Expected ports:

- visual expert: `http://localhost:8002/predict`
- BLIP search expert: `http://localhost:8102/attention_map`
- SAM2: `http://127.0.0.1:8202/sam2/point_predict`

### Required Code Configuration for GPU 2 Services

The startup scripts become GPU-2 aware, but several clients are hard-coded to
the README's GPU-0 ports. Update these before running on GPU 2 services.

1. In `code/src/policies/vstar/MCTS.py`, change the visual expert ports:

```python
self.expert_ports = [2]
self.expert_ports = [port + 8000 for port in self.expert_ports]
```

This makes `get_expert_boxes()` call `http://localhost:8002/predict`.

2. In `code/src/policies/vstar/client.py`, change defaults:

```python
endpoint: str = "http://localhost:8102/attention_map"
endpoint: str = "http://127.0.0.1:8202/sam2/point_predict"
```

3. In `code/src/policies/vstar/visual_grounding.py`, change hard-coded calls:

```python
sam_endpoint: str = "http://127.0.0.1:8202/sam2/point_predict"
resized_img, heatmap = get_heatmap(
    img,
    question,
    endpoint="http://localhost:8102/attention_map",
    block=BLOCK,
)
```

Without these changes, the main process will still try to contact ports
`8000`, `8100`, and `8200`.

### Required Code Configuration for GPU 3 Main Runtime

`code/src/qwen_runtime.py` currently contains this line at import time:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```

For physical GPU 3, change it to:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
```

A more adaptable local edit is:

```python
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
```

With `setdefault`, the shell command controls the GPU and the file only provides
a fallback. This is safer if you later run with different devices.

Then run the main evaluation with GPU 3:

```bash
CUDA_VISIBLE_DEVICES=3 bash code/scripts/vstar/stream_vstar_qwen.sh oursmcts False
```

Also update `CKPT` and `QUESTION_FILE` inside
`code/scripts/vstar/stream_vstar_qwen.sh` for your local machine.

### Single-Sample GPU 2/3 Check

After the three services are running on GPU 2 and the client ports are updated,
test one example on GPU 3:

```bash
CUDA_VISIBLE_DEVICES=3 python code/src/run_single_sample.py \
  --question-file playground/data/eval/vstar/test_questions.tsv \
  --row-idx 0 \
  --answers-file /tmp/deepscan_one.jsonl \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --method_name vstar.oursmcts \
  --image-size 1024 \
  --temperature 0
```

This verifies the full chain:

1. Qwen runtime loads on GPU 3.
2. BLIP heatmap service responds on port `8102`.
3. SAM2 point-prompt service responds on port `8202`.
4. LangSAM visual expert responds on port `8002`.
5. A JSONL prediction is written.

## Port Adaptability Notes

The most important implementation detail is that ports are not configured in one
central place.

Current port logic:

- `expert_server/start_server.sh`: `8000 + physical_gpu_id`
- `blip_server/start_server.sh`: `8100 + physical_gpu_id`
- `sam2_server/start_server.sh`: `8200 + physical_gpu_id`
- `MCTS.py`: hard-coded visual expert list starts at `[0]`, therefore `8000`
- `client.py`: hard-coded BLIP and SAM2 defaults, therefore `8100` and `8200`
- `visual_grounding.py`: hard-coded BLIP and SAM2 URLs, therefore `8100` and
  `8200`
- `qwen_runtime.py`: hard-coded main CUDA device, currently `"1"`

For repeated experiments, consider replacing the hard-coded URLs with
environment variables such as:

```bash
DEEPSCAN_VISUAL_EXPERT_URL=http://localhost:8002/predict
DEEPSCAN_BLIP_URL=http://localhost:8102/attention_map
DEEPSCAN_SAM2_URL=http://127.0.0.1:8202/sam2/point_predict
```

That would make the code independent of physical GPU IDs and would avoid editing
Python files when moving from GPUs 0/1 to GPUs 2/3.

## Checkpoints and Local Paths

The code currently requires local path edits for model assets:

- `code/scripts/blip_server/blip_service.py`
  - `LOCAL_TOKENIZER_PATH`
- `code/scripts/expert_server/model_service.py`
  - optional offline `LangSAM(...)` checkpoint paths for SAM and GroundingDINO
- `code/scripts/sam2_server/sam2_service.py`
  - `SAM2_REPO_ROOT`
  - `--ckpt` default
- `code/scripts/vstar/stream_vstar_qwen.sh`
  - `CKPT`
  - `QUESTION_FILE`

These are environment-specific and should be checked before every new machine or
conda environment.

## Practical Debugging Checklist

Before launching the full V* run:

1. Confirm each service logs the expected GPU-2 port: `8002`, `8102`, `8202`.
2. Confirm the client URLs in `MCTS.py`, `client.py`, and
   `visual_grounding.py` match those ports.
3. Confirm `qwen_runtime.py` no longer forces GPU 1.
4. Run `run_single_sample.py` with `CUDA_VISIBLE_DEVICES=3`.
5. Inspect `/tmp/deepscan_one.jsonl` and verify `text` contains a predicted
   option letter.
6. Launch `stream_vstar_qwen.sh` after `CKPT` and `QUESTION_FILE` are correct.

