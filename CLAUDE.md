# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DeepScan is a **training-free** test-time framework for visually grounded reasoning in LVLMs (CVPR 2026). It runs a bottom-up pipeline — **Hierarchical Scanning → Refocusing → Evidence-Enhanced Reasoning** — by orchestrating one main LVLM process against three local expert HTTP services. There is no training loop; all models are frozen and used at inference time.

## Multi-environment / multi-service setup

This is **not** a single-process app. Running anything end-to-end requires **three conda envs** and **three running services** plus the main runtime, typically split across GPUs:

| Env | Purpose | Service(s) it hosts |
| --- | --- | --- |
| `lavis` | search expert | BLIP Grad-CAM (`blip_server`) |
| `langsam` | visual expert + segmentation | LangSAM/GroundingDINO (`expert_server`), SAM2 (`sam2_server`) |
| `deepscan` | main LVLM runtime | `run.py` / Qwen-VL |

The `lavis` env additionally requires a **monkey patch** to LAVIS `blip_image_text_matching.py` (replace `self.tokenizer.enc_token_id` with `self.tokenizer.convert_tokens_to_ids("[ENC]")`) — see README "Required Monkey Patch". Without it the BLIP search expert fails.

## Common commands

Start the three expert services (each in its env, on the expert GPU):
```bash
conda activate langsam && bash code/scripts/expert_server/start_server.sh   # LangSAM /predict
conda activate lavis   && bash code/scripts/blip_server/start_server.sh     # BLIP /attention_map
conda activate langsam && bash code/scripts/sam2_server/start_server.sh     # SAM2 /sam2/point_predict
```

Full V* evaluation (main runtime), which shards across visible GPUs, merges shard JSONLs, and runs the dataset evaluator:
```bash
conda activate deepscan
bash code/scripts/vstar/stream_vstar_qwen.sh oursmcts False
# arg1 = METHOD (must be `oursmcts` for DeepScan; default `clean` is not a registered V* policy)
# arg2 = DEBUG
```

Run / debug a **single sample** (the fastest way to exercise the full chain):
```bash
python code/src/run_single_sample.py \
  --question-file playground/data/eval/vstar/test_questions.tsv \
  --row-idx 0 \                 # or --index N to select by the TSV `index` column
  --answers-file /tmp/out.jsonl \
  --model-path <ckpt> \
  --method_name vstar.oursmcts \
  --image-size 1024
```

`run.py --debug True` runs one random sample **and** starts `debugpy` on port 5678, blocking until a debugger attaches.

There is no build, lint, or test suite. Verification is empirical: run a single sample, inspect the JSONL, and confirm `text` contains a predicted option letter.

## Architecture (the parts that span files)

- **Entry point `code/src/run.py` is method-agnostic.** It loads a TSV, shards it via `--num-chunks`/`--chunk-idx` (contiguous split, see `utils.get_chunk`), and dispatches each row to a `QuestionSample` subclass resolved from `policies.policy_map[--method_name]`. The DeepScan algorithm lives entirely in the policy class, not in `run.py`.
- **Policy registration is by convention, not explicit.** `policies/vstar/__init__.py` imports every file in the dir, finds `QuestionSample` subclasses, strips the `QuestionSample` suffix, and lowercases it; `policies/__init__.py` then prefixes `vstar.`. So `OursMCTSQuestionSample` → method name `vstar.oursmcts`. Adding a new policy = adding a class with that suffix.
- **The three pipeline stages map to specific files:**
  - *Hierarchical Scanning*: `visual_grounding.py` (`grounding()`) calls BLIP `/attention_map`, Otsu-thresholds the heatmap into connected-component centroids, feeds them to SAM2 `/sam2/point_predict`, and merges masks into candidate crops.
  - *Refocusing*: `ours_mcts.py` / `MCTS.py` — an MCTS search (`max_depth=3`, `n_simulations=6`, actions `repeat_question` and `zoom_out`) over crops, calling the LangSAM `/predict` visual expert. Reward favors the smallest crop that still contains all extracted key objects (`1 - valid_area_ratio`).
  - *Evidence-Enhanced Reasoning*: `MCTS.py` / `policy.py` build a multi-image evidence prompt; with `use_ensemble`, node answers are weighted by leaf reward and the top vote wins (falls back to direct inference on the original image if all weights are zero).
- **LVLM inference**: local HF inference via `code/src/qwen_runtime.py` (supports Qwen2.5-VL and Qwen3-VL, which take **different evidence-image layouts** in the final prompt — see `docs/architecture.md` §6). `utils.get_openai_clients_and_models()` is an alternative path targeting an OpenAI-compatible server.
- **Inter-service payloads** are base64-encoded image strings throughout (`client.py`, `blip_service.py`, `sam2_service.py`).

## Ports & hardcoded paths — the main footgun

Ports are **GPU-derived and not centralized**, so moving GPUs requires editing multiple files:
- Servers compute ports from `CUDA_VISIBLE_DEVICES`: visual expert `8000+gpu`, BLIP `8100+gpu`, SAM2 `8200+gpu` (in the `start_server.sh` scripts).
- Clients **hardcode** the GPU-0 ports: `MCTS.py` (`expert_ports`), `client.py` (BLIP + SAM2 defaults), `visual_grounding.py` (BLIP + SAM2 URLs). These must be changed to match if the expert GPU isn't 0.
- `qwen_runtime.py` hardcodes the main runtime's CUDA device at import time (`os.environ["CUDA_VISIBLE_DEVICES"] = ...`).

Other machine-specific paths to update on a new setup: `CKPT` and `QUESTION_FILE` in `stream_vstar_qwen.sh`; `LOCAL_TOKENIZER_PATH` in `blip_service.py`; `LangSAM(...)` checkpoint paths in `model_service.py`; `SAM2_REPO_ROOT` and `--ckpt` default in `sam2_service.py` (keep `--cfg` package-relative — Hydra won't resolve an absolute path). The SAM2 base-plus checkpoint must be placed under the installed `sam2` package's `checkpoints/` dir in the `langsam` env.

`download_models.py` fetches the required HF checkpoints into `models/` (gitignored, along with `playground/`, `eval/`, `tmp/`).

## Further reading

The repo ships detailed living docs — consult them before deep changes:
- `docs/architecture.md` — paper-concept → file map, full method flow, and a worked GPU-2/3 reconfiguration walkthrough.
- `docs/experiment_flow.md` — exact `run.py` execution path, sharding, result JSONL schema, and failure/empty-output triage.
- `README.md` — environment setup, model download, and service launch logs.
