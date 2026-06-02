# Experiment Flow in `code/src/run.py`

This document explains how the batch experiment runner works in the current
DeepScan implementation. It follows the real code path from the shell script,
through `run.py`, into the policy implementation, and finally into the JSONL
prediction file.

## High-Level Flow

The experiment flow has two layers:

1. `code/scripts/vstar/stream_vstar_qwen.sh` launches one or more `run.py`
   processes, one per visible GPU.
2. `code/src/run.py` loads one dataset shard, instantiates the selected policy,
   processes each sample asynchronously, and writes predictions to JSONL.

`run.py` itself is method-agnostic. It does not contain the DeepScan algorithm.
Instead, it dispatches to a policy class through `policy_map`. For DeepScan on
V*, the method name is:

```text
vstar.oursmcts
```

That resolves to `OursMCTSQuestionSample` under `code/src/policies/vstar/`.

## Shell Launcher: `stream_vstar_qwen.sh`

The common V* experiment entry point is:

```bash
bash code/scripts/vstar/stream_vstar_qwen.sh oursmcts False
```

In `code/scripts/vstar/stream_vstar_qwen.sh`, the first argument becomes
`METHOD`:

```bash
METHOD=${1:-clean}
DEBUG=${2:-False}
```

For DeepScan, `METHOD` should be `oursmcts`. The script later constructs:

```bash
--method_name "${DATASET_FORMAT}.${METHOD}"
```

With `DATASET_FORMAT="vstar"`, this becomes:

```text
vstar.oursmcts
```

The script gets the visible GPUs from `CUDA_VISIBLE_DEVICES`:

```bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
```

If `CUDA_VISIBLE_DEVICES=2,3`, then `CHUNKS=2`. The script starts two
background processes:

```bash
CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python code/src/run.py ...
```

Each process receives:

- the same question TSV
- a different `--chunk-idx`
- the same `--num-chunks`
- a different output shard path

After all background processes finish, the script concatenates shard files into
one merged JSONL:

```bash
output_file="$OUT_DIR/${NAME}.jsonl"
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$OUT_DIR/${CHUNKS}_${IDX}_${NAME}.jsonl" >> "$output_file"
done
```

Then it calls the dataset evaluator:

```bash
python "/root/autodl-tmp/playground/data/eval/$DATASET/eval.py" \
    --path "$output_file"
```

## `run.py` Arguments

`code/src/run.py` defines the batch runner CLI. The most important arguments
are:

| Argument | Role |
| --- | --- |
| `--model-path` | Local or Hugging Face model path passed into the policy and LVLM runtime |
| `--question-file` | TSV file loaded by `pandas.read_table()` |
| `--answers-file` | JSONL output path for this process or shard |
| `--num-chunks` | Total number of dataset shards |
| `--chunk-idx` | Which shard this process handles |
| `--method_name` | Policy key looked up in `policy_map`, for example `vstar.oursmcts` |
| `--image-size` | Max image size used by the API-style generation helper |
| `--temperature` | Generation temperature passed through to model calls |
| `--single-pred-prompt` | Adds a direct multiple-choice instruction to the final prompt |
| `--all-rounds` | Optional mode that creates one round per answer option |
| `--debug` | Runs a randomly selected sample and opens `debugpy` |

The default `--method_name` is `common`, but that policy is not registered in
the current V* policy map. For DeepScan experiments, use `vstar.oursmcts`.

## Dataset Loading and Sharding

The main function is `eval_model(args)`.

It starts by loading the TSV:

```python
questions = pd.read_table(os.path.expanduser(args.question_file))
```

Then it keeps only one shard:

```python
questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
```

The sharding helpers are in `code/src/utils.py`:

```python
def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
```

This means sharding is contiguous, not round-robin. If the TSV has 191 rows and
`--num-chunks 2`, shard 0 receives the first roughly half of the rows and shard
1 receives the remaining rows.

The code then prepares the output directory:

```python
answers_file = os.path.expanduser(args.answers_file)
os.makedirs(os.path.dirname(answers_file), exist_ok=True)
```

One practical detail: `answers_file` should include a directory component. A
plain filename like `answer.jsonl` gives `os.path.dirname("answer.jsonl") == ""`,
which can make `os.makedirs("")` fail. The shell script avoids this by passing a
full path under `eval/...`.

## Sample Argument Creation

After sharding, the DataFrame is converted into row dictionaries:

```python
rows_as_dicts = questions.to_dict(orient="records")
```

For each row, `run.py` creates one or more sample jobs:

```python
if args.all_rounds:
    num_rounds = len(get_options(row, ['A', 'B', 'C', 'D']))
else:
    num_rounds = 1

for round_idx in range(num_rounds):
    sample_args.append((row, args, round_idx))
```

In normal DeepScan runs, `--all-rounds` is not used, so each TSV row produces
one sample. If `--all-rounds` is enabled, the number of rounds equals the number
of non-empty multiple-choice options found by `get_options()`.

## Policy Dispatch

Sample construction happens in `create_sample()`:

```python
async def create_sample(args):
    row, method_args, round_idx = args
    QuestionSample = policy_map[method_args.method_name]
    return QuestionSample(row, method_args, round_idx)
```

`policy_map` comes from `code/src/policies/__init__.py`.

That file imports the V* policy map and prefixes every V* policy name with
`vstar.`:

```python
for name, cls in vstar_policy_map.items():
    policy_map[f"vstar.{name}"] = cls
```

The V* policy map is built dynamically in `code/src/policies/vstar/__init__.py`.
It imports every Python file in that directory, finds subclasses of
`QuestionSample`, removes the `QuestionSample` suffix, lowercases the result,
and registers it. Therefore:

```text
OursMCTSQuestionSample -> oursmcts -> vstar.oursmcts
```

That is how the CLI argument reaches the DeepScan policy class.

## Async Sample Creation

`run.py` creates all sample objects through coroutines:

```python
tasks = [create_sample(args) for args in sample_args]
for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Creating samples"):
    sample = await coro
    samples.append(sample)
```

This phase instantiates policy objects. For `vstar.oursmcts`, construction
eventually calls `MCTSQuestionSample.__init__()`, which decodes the base64 image
to get image dimensions, initializes MCTS parameters, and prepares expert
service ports.

The sample objects are not processed yet in this phase.

## Async Sample Processing

After construction, `run.py` processes each sample:

```python
tasks = [process_sample(sample) for sample in samples]
for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing samples"):
    result = await coro
    results.append(result)
```

`process_sample()` simply calls:

```python
return await sample.process()
```

The base `QuestionSample.process()` method in
`code/src/policies/vstar/policy.py` wraps the policy-specific `_process()` in a
try/except block. If a sample crashes, it returns a fallback result with:

```python
"text": "A"
```

and includes error metadata.

For `vstar.oursmcts`, `_process()` is implemented through
`MCTSQuestionSample._process()` and `OursMCTSQuestionSample.get_final_answer()`.
The policy-level flow is:

1. Extract key objects from the question using the LVLM.
2. Run DeepScan grounding and evidence proposal.
3. Justify candidate evidence crops.
4. Run the MCTS refocusing search.
5. Query the LVLM over the selected evidence images.
6. Vote over node answers using MCTS rewards.
7. Return a result dictionary.

## Result Format

Each policy returns one dictionary. For `vstar.oursmcts`, the result is:

```python
{
    "question_id": self.row['index'],
    "round_id": self.round_idx,
    "prompt": prompt,
    "text": final_answer,
    "options": self.options,
    "option_char": self.cur_option_char,
    "answer_id": shortuuid.uuid(),
    "model_id": self.args.model_path,
    "answer": self.row['answer'],
}
```

The most important fields are:

- `question_id`: the TSV row's `index`
- `prompt`: the final multiple-choice prompt sent to the LVLM
- `text`: the predicted option letter
- `answer`: the gold option letter
- `model_id`: the model path passed through `--model-path`

If an exception occurs, the fallback result has the same core fields plus
`metadata` containing the error string and traceback.

## JSONL Writing

After all samples finish, `run.py` writes the result list:

```python
with open(answers_file, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")
```

The output file is overwritten on each run. In the shell launcher, each process
writes a separate shard file. The launcher then overwrites and rebuilds the
merged file.

For example, with two chunks and Qwen3-VL:

```text
eval/vstar/deepscan/Qwen3-VL-8B-Instruct/test_questions/oursmcts/
├── 2_0_Qwen3-VL-8B-Instruct.jsonl
├── 2_1_Qwen3-VL-8B-Instruct.jsonl
└── Qwen3-VL-8B-Instruct.jsonl
```

The first two files are generated by `run.py`; the final merged file is generated
by `stream_vstar_qwen.sh`.

## Debug Mode

`run.py` has two debug behaviors.

First, if `--debug True`, it starts `debugpy`:

```python
debugpy.listen(5678)
debugpy.wait_for_client()
debugpy.breakpoint()
```

Second, inside `eval_model()`, debug mode reduces the workload to one random
sample:

```python
if args.debug:
    sample_args = [random.choice(sample_args)]
```

If `--debug False`, the code sets `random.seed(42)`, but it does not shuffle or
sample the dataset in the current implementation.

## Single-Sample Runner vs Batch Runner

`code/src/run_single_sample.py` is separate from `run.py`.

Use `run_single_sample.py` when you want to select one TSV row by `--index` or
`--row-idx` and write one JSONL line to a specific file such as `/tmp/out.jsonl`.

Use `run.py` or `stream_vstar_qwen.sh` when you want batch evaluation,
chunk-based data parallelism, merged outputs, and evaluator execution.

This distinction matters because a command like:

```bash
python code/src/run_single_sample.py \
  --answers-file /tmp/out.jsonl \
  ...
```

does not write to the merged evaluation file under `eval/...`. Only
`stream_vstar_qwen.sh` creates or updates that merged file.

## Failure and Empty-Output Checks

If an expected output file is empty, check which layer created it.

- If a shard file is empty, inspect the corresponding `run.py` process logs.
- If the merged file is empty, check whether the shard files exist and contain
  JSONL lines before the script concatenates them.
- If `run_single_sample.py` succeeds but the merged eval file is empty, that is
  expected: the single-sample command writes only to its `--answers-file`.
- If a result contains `"text": "A"` with error metadata, the policy crashed and
  `QuestionSample.process()` returned the fallback answer.

Useful paths:

- Batch shard output: `--answers-file` passed to `run.py`
- Batch merged output: `output_file="$OUT_DIR/${NAME}.jsonl"` in
  `stream_vstar_qwen.sh`
- Single-sample output: `--answers-file` passed to `run_single_sample.py`

