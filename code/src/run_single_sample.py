import argparse
import asyncio
import json
import os

import pandas as pd

from utils import get_options, is_none
from policies import policy_map


def _select_row(questions: pd.DataFrame, args: argparse.Namespace) -> dict:
    if args.row_idx is not None:
        if args.row_idx < 0 or args.row_idx >= len(questions):
            raise ValueError(
                f"--row-idx {args.row_idx} is out of range (0..{len(questions) - 1})."
            )
        return questions.iloc[args.row_idx].to_dict()

    if args.index is None:
        raise ValueError("Provide --index or --row-idx to select a single sample.")

    if "index" not in questions.columns:
        raise ValueError(
            "The question file has no 'index' column. Use --row-idx to select a row."
        )

    matches = questions[questions["index"] == args.index]
    if matches.empty:
        raise ValueError(f"No sample found with index={args.index}.")
    if len(matches) > 1:
        print(f"Warning: multiple rows matched index={args.index}; using the first match.")
    return matches.iloc[0].to_dict()


def _ensure_image_base64(row: dict) -> None:
    if "image" not in row or is_none(row["image"]):
        raise ValueError(
            "Missing base64 image in the question file. "
            "Use the TSV dataset with the 'image' column populated."
        )


async def run_single_sample(args: argparse.Namespace) -> None:
    questions = pd.read_table(os.path.expanduser(args.question_file))
    print(f"[trace:single] loaded {len(questions)} rows from {args.question_file}")
    row = _select_row(questions, args)
    _ensure_image_base64(row)
    print(
        "[trace:single] selected "
        f"index={row.get('index')} category={row.get('category')} "
        f"question={row.get('question')!r}"
    )

    if args.method_name not in policy_map:
        raise ValueError(f"Unknown method_name: {args.method_name}")

    if args.all_rounds:
        num_rounds = len(get_options(row, ["A", "B", "C", "D"]))
    else:
        num_rounds = 1

    QuestionSample = policy_map[args.method_name]
    results = []
    for round_idx in range(num_rounds):
        print(f"[trace:single] processing round_idx={round_idx} method={args.method_name}")
        sample = QuestionSample(row, args, round_idx)
        result = await sample.process()
        print(
            "[trace:single] result "
            f"question_id={result.get('question_id')} text={result.get('text')!r} "
            f"answer={result.get('answer')!r}"
        )
        results.append(result)

    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    with open(answers_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"[trace:single] wrote {len(results)} result(s) to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=" /root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument(
        "--question-file",
        type=str,
        default="playground/data/eval/vstar/test_questions.tsv",
    )
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--method_name", type=str, default="vstar.oursmcts")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--index", type=int, help="Select sample by the 'index' column.")
    parser.add_argument(
        "--row-idx",
        type=int,
        default=None,
        help="Select sample by row order (0-based).",
    )

    def str2bool(v):
        return v.lower() == "true"

    parser.add_argument("--debug", type=str2bool, help="debug mode", default=False)
    args = parser.parse_args()

    if args.debug:
        import debugpy

        debugpy.listen(5678)
        print("Waiting for debugpy connection...")
        debugpy.wait_for_client()
        print("Breakpoint stopped here, ready for debugging...")
        debugpy.breakpoint()

    asyncio.run(run_single_sample(args))
