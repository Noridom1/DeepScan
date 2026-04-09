<div align="center">
  <h1>🔎 DeepScan: A Training-Free Framework for Visually Grounded Reasoning in Large Vision-Language Models</h1>
  <p>
    <em>Official implementation of the CVPR 2026 paper</em>
  </p>

  <p>
    <a href="https://arxiv.org/abs/2603.03857"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%202026-1D4ED8"></a>
    <a href="https://arxiv.org/abs/2603.03857"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.03857-B31B1B"></a>
    <a href="#"><img alt="Framework" src="https://img.shields.io/badge/Framework-Training--Free-10B981"></a>
    <a href="#"><img alt="Grounding" src="https://img.shields.io/badge/Grounding-Bottom--Up-111827"></a>
  </p>
</div>

> **TL;DR.** DeepScan is a **training-free** framework for **visually grounded reasoning** in LVLMs. Instead of relying on brittle one-shot, coarse-to-fine localization, it adopts a **bottom-up** pipeline: **Hierarchical Scanning** for cue discovery and evidence recovery, **Refocusing** for context-optimal evidence views, and **Evidence-Enhanced Reasoning** for final answer generation from multi-granular evidence memory.

---

## 🔥 News
- [ ] Release the evaluation scripts.
- [x] **2026-03.** The core codebase is open-sourced.
- [x] **2026-02.** DeepScan was accepted to **CVPR 2026** main track.

---

## 👀 Overview
Humans often solve challenging visual problems in a **bottom-up** manner: they first identify subtle local cues, then recover the full evidence from those cues, and finally reason over the recovered evidence. DeepScan is built on the same intuition.

DeepScan contains three tightly coupled stages:

1. **Hierarchical Scanning**
   - Partition the image into local patches.
   - Use a **search expert** to produce patch-wise attention maps.
   - Convert connected cue regions into **point-based proxies** using both **semantic saliency** and **topological interiority**.
   - Recover image-level evidence via **point-prompt segmentation**, followed by morphological post-processing.
   - Retain only the **top-k smallest evidence candidates** for efficient evidence judgment.

2. **Refocusing**
   - Starting from the fused evidence crop, search over a concise set of candidate views.
   - Use **Zoom-In** and **Zoom-Out** actions to calibrate the surrounding context.
   - Select the **smallest view that still fully contains the evidence needed for answering**.

3. **Evidence-Enhanced Reasoning**
   - Build a **Hybrid Evidence Memory** composed of:
     - **fine-grained evidence crops** from Hierarchical Scanning, and
     - a **coarse-grained refined view** from Refocusing.
   - Materialize them as an ordered multi-image prompt for the LVLM.
   - Generate answers that are both **more accurate** and **better grounded** in the visual evidence.

Unlike RL-based visually grounded reasoning methods, DeepScan is **plug-and-play** and **training-free**. It can be integrated with different LVLM backbones without additional adaptation cost.

---

## 🏗 Repository Structure
```text
DeepScan/
├── code/
│   ├── scripts/
│   │   ├── blip_server/       # Search-expert service
│   │   ├── expert_server/     # Visual-expert service
│   │   ├── sam2_server/       # SAM2 segmentation service
│   │   ├── lmm_server/        # LVLM serving / runtime scripts
│   │   ├── pope/
│   │   └── vstar/
│   └── src/
│       ├── eval.py
│       ├── qwen_runtime.py
│       ├── run.py
│       ├── utils.py
│       └── policies/
├── lavis.yml
├── langsam.yml
├── dyfo.yml
└── README.md
🛠 Environment Setup

DeepScan uses three separate environments:

lavis for the search expert
langsam for the visual expert and SAM2
deepscan for the LVLM runtime / main pipeline
1) Clone the repository
git clone https://github.com/YChenL/DeepScan
cd DeepScan
2) Create the environments
Search expert environment
conda env create -f lavis.yml
conda activate lavis
Visual expert environment
conda env create -f langsam.yml
conda activate langsam
DeepScan runtime environment
conda env create -f dyfo.yml
conda activate deepscan
🔧 Required Monkey Patch for LAVIS

After creating the lavis environment, please patch the following file:

your_env_path/lavis/models/blip_models/blip_image_text_matching.py

Replace the following line at Line 76, Line 109, and Line 122:

# encoder_input_ids[:, 0] = self.tokenizer.enc_token_id  # extra code
encoder_input_ids[:, 0] = self.tokenizer.convert_tokens_to_ids("[ENC]")

In other words, if the original code uses:

encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

please replace it with:

encoder_input_ids[:, 0] = self.tokenizer.convert_tokens_to_ids("[ENC]")

This patch is required for the BLIP-based search expert used in our pipeline.

📦 Model Preparation

Please download the following checkpoints from Hugging Face:

BERT tokenizer / backbone
google-bert/bert-base-uncased
GroundingDINO
IDEA-Research/grounding-dino-base
SAM2
facebook/sam2.1-hiera-small
facebook/sam2.1-hiera-base-plus
LVLM backbone
Qwen/Qwen3-VL-8B-Instruct
Example
huggingface-cli download google-bert/bert-base-uncased \
    --local-dir bert-base-uncased \
    --local-dir-use-symlinks False \
    --resume-download

You can download the other checkpoints in the same way by replacing the model name and local directory.

Additional requirement for SAM2

Please place the checkpoint from:

facebook/sam2.1-hiera-base-plus

into the checkpoints directory of the sam2 package inside the langsam environment.

🛣 Path Configuration

Before launching the system, you need to replace several environment-specific local paths in the code.

1) BLIP tokenizer path

File:

DeepScan/code/scripts/blip_server/blip_service.py

Set:

LOCAL_TOKENIZER_PATH = "your/local/google-bert/bert-base-uncased/path"
2) LangSAM / GroundingDINO / SAM2 checkpoint paths

File:

DeepScan/code/scripts/expert_server/model_service.py

Set the checkpoint paths in the LangSAM(...) initialization, e.g.

self.model = LangSAM(
    sam_type="sam2.1_hiera_small",
    ckpt_path_sam="/your/local/facebook/sam2.1-hiera-small/sam2.1_hiera_small.pt",
    ckpt_path_gdino="/your/local/IDEA-Research/grounding-dino-base"
)
3) SAM2 repository root

File:

DeepScan/code/scripts/sam2_server/sam2_service.py

Set:

SAM2_REPO_ROOT = Path("/your/envs/langsam/lib/python3.11/site-packages/sam2")

This should be the absolute path to the installed sam2 package inside your langsam environment.

📂 Dataset Preparation

Please download the evaluation datasets from:

oking0197/Dyfo

After extraction, organize them as:

DeepScan/
├── code
│   ├── scripts
│   └── src
└── playground
    └── data
        └── eval
            ├── vstar
            └── ...

In other words, the benchmark folders should be placed under:

playground/data/eval/
🧠 Experts and Backbones

DeepScan augments an LVLM with two plug-and-play experts:

A) Search Expert

The paper uses BLIP-ITM as the search expert to produce patch-wise Grad-CAM attention maps for local cue exploration.

B) Visual Expert

The visual expert exposes two primitives:

point-prompt segmentation
text-conditioned detection

In our implementation, the visual grounding pipeline is realized via:

a LangSAM-based detection service
a SAM2 point-prompt segmentation service
C) LVLM Backbone

For local inference, you may serve a compatible LVLM backend such as:

Qwen3-VL-8B-Instruct

The repository also includes example scripts for Qwen-style runtime under code/scripts/.

🚀 Launch the Servers

DeepScan is a multi-service pipeline. In a typical setup, you should start:

the visual-expert server
the search-expert server
the SAM2 segmentation server

For example, if you have one GPU reserved for expert services (e.g. cuda:0), you can launch them in separate terminals as follows:

On cuda:0
Visual expert
conda activate langsam
bash DeepScan/code/scripts/expert_server/start_server.sh
Search expert
conda activate lavis
bash DeepScan/code/scripts/blip_server/start_server.sh
SAM2 server
conda activate langsam
bash DeepScan/code/scripts/sam2_server/start_server.sh

Note: Please adjust CUDA device assignment, ports, and checkpoint paths in the corresponding scripts before launch.

▶️ Run DeepScan

After the expert services are running, use the deepscan environment to launch the main pipeline.

For example, if your LVLM inference runs on cuda:1 or other remaining GPUs:

conda activate deepscan
bash DeepScan/code/scripts/vstar/stream_vstar_qwen.sh

You may also invoke the main entry point directly through code/src/run.py, depending on your local setup.

🧩 Prompting Roles in DeepScan

DeepScan relies on three lightweight LVLM query templates:

Evidence Decomposition
Extract the objects mentioned in the question.
Used to decide whether the question is single-object or multi-object, and thus which patch size to use.
Evidence Judgment
Judge whether a cropped evidence candidate actually contains clues for answering the question.
View Completeness Justification
Judge whether a refocused view fully contains every target object without truncation.
📈 Results at a Glance

DeepScan provides strong gains on fine-grained and visually grounded reasoning benchmarks.

V* (Qwen2.5-VL-7B backbone): 90.6% overall
93.0% Attribute
86.8% Spatial
Improvement over vanilla Qwen2.5-VL-7B:
+16.3% on V*
+5.5% on TreeBench
HR-Bench:
75.0% on HR-4K
72.4% on HR-8K
TreeBench:
42.5% overall
37.3 mIoU
Scaling:
DeepScan-72B reaches 94.2% on V* at k = ∞

DeepScan is also competitive with strong visually grounded reasoning methods while remaining fully training-free.

⚡ Efficiency Notes

DeepScan is a test-time scaling framework, so it introduces extra inference cost compared with vanilla one-shot inference. At the same time, it admits an explicit performance-efficiency trade-off through:

the patch size
the number of retained evidence candidates (k)
batched engineering optimizations

In the optimized implementation discussed in the supplementary material, DeepScan benefits substantially from:

batched attention-map computation
batched top-k evidence judgment
batched view justification
vLLM-based serving

These optimizations reduce the sequential overhead of visually grounded search and significantly improve throughput.

🙏 Acknowledgements

DeepScan builds on several excellent open-source projects and model ecosystems. We would like to give special thanks first to DyFo
 for its inspiring open-source release. We also acknowledge the following projects and model ecosystems:

Qwen2-VL / Qwen2.5-VL / Qwen3-VL
LAVIS
LangSAM
SAM2
vLLM

We thank the authors and maintainers of these projects for making their work available.

📜 Citation

If you find DeepScan useful, please cite:

@article{li2026deepscan,
  title={DeepScan: A Training-Free Framework for Visually Grounded Reasoning in Large Vision-Language Models},
  author={Li, Yangfu and Zhan, Hongjian and Chen, Jiawei and Gong, Yuning and Liu, Qi and Lu, Yue},
  journal={arXiv preprint arXiv:2603.03857},
  year={2026}
}