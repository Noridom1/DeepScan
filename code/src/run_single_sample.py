import argparse
import asyncio
import json
import os

import pandas as pd

from log import configure as log_configure, get_logger
from utils import get_options, is_none
from policies import policy_map

logger = get_logger("single")


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
        logger.warning("multiple rows matched index=%s; using the first match.", args.index)
    return matches.iloc[0].to_dict()


def _ensure_image_base64(row: dict) -> None:
    if "image" not in row or is_none(row["image"]):
        raise ValueError(
            "Missing base64 image in the question file. "
            "Use the TSV dataset with the 'image' column populated."
        )


async def run_single_sample(args: argparse.Namespace) -> None:
    questions = pd.read_table(os.path.expanduser(args.question_file))
    logger.info("loaded %d rows from %s", len(questions), args.question_file)
    row = _select_row(questions, args)
    _ensure_image_base64(row)
    logger.info(
        "selected index=%s category=%s question=%r",
        row.get("index"), row.get("category"), row.get("question"),
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
        logger.info("processing round_idx=%d method=%s", round_idx, args.method_name)
        # Set up round-specific artifact directory if all-rounds is enabled
        if args.save_artifacts and args.all_rounds and num_rounds > 1:
            args.artifact_run_dir = os.path.join(args.artifact_dir_base, f"round-{round_idx}")
            os.makedirs(args.artifact_run_dir, exist_ok=True)
        elif args.save_artifacts:
            args.artifact_run_dir = args.artifact_dir_base
        sample = QuestionSample(row, args, round_idx)
        result = await sample.process()
        logger.info(
            "result question_id=%s text=%r answer=%r",
            result.get("question_id"), result.get("text"), result.get("answer"),
        )
        results.append(result)

    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    with open(answers_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info("wrote %d result(s) to %s", len(results), answers_file)


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
    parser.add_argument("--save-artifacts", type=str2bool, help="save artifacts", default=False)
    parser.add_argument("--artifact-dir", type=str, default="artifacts", help="directory to save artifacts")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["debug", "info", "warning", "error", "off"],
        help="Logging verbosity (default: info, or DEEPSCAN_LOG_LEVEL env var).",
    )
    args = parser.parse_args()

    log_configure(args.log_level)

    # Set up artifact directory if enabled
    if args.save_artifacts:
        row = _select_row(pd.read_table(os.path.expanduser(args.question_file)), args)
        sample_id = row.get("index") if "index" in row and row.get("index") is not None else args.row_idx
        artifact_run_dir = os.path.expanduser(args.artifact_dir)
        artifact_run_dir = os.path.join(artifact_run_dir, str(sample_id))
        # Note: round-specific subdirectory will be appended by sample if --all-rounds is used
        args.artifact_dir_base = artifact_run_dir
        args.artifact_run_dir = artifact_run_dir  # Will be updated per round if needed
    else:
        args.artifact_run_dir = None
        args.artifact_dir_base = None

    if args.debug:
        import debugpy

        debugpy.listen(5678)
        logger.info("waiting for debugpy connection on port 5678")
        debugpy.wait_for_client()
        logger.info("debugger attached, resuming")
        debugpy.breakpoint()

    asyncio.run(run_single_sample(args))
