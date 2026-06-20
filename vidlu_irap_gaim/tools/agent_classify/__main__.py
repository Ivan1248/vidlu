"""
Single frontend for agent-based iRAP attribute classification.

The workflow has two phases with the agent running externally in between, so this
exposes two subcommands:

    # 1. Build the task list + prompt for the agent.
    python -m vidlu_irap_gaim.tools.agent_classify prepare \\
        --dataset bih --split test --output-dir ./agent_run/

    # 2. (the agent reads the prompt, classifies, and writes
    #     agent_predictions_bih_test.json into ./agent_run/)

    # 3. Score the agent's predictions against ground truth. `prepare` prints this
    #    exact command, with --predictions pointing at the file the agent wrote.
    python -m vidlu_irap_gaim.tools.agent_classify evaluate \\
        --dataset bih --split test \\
        --predictions ./agent_run/agent_predictions_bih_test.json

Artifact file names are defined once in ``_common.agent_artifact_paths``.
"""

import argparse

from .prepare import prepare_agent_tasks
from .evaluate import evaluate_agent_predictions


def _cmd_prepare(args):
    from ._common import agent_artifact_paths

    result = prepare_agent_tasks(
        dataset_name=args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        limit=args.limit,
        prompt_config=args.prompt_config,
    )
    predictions = agent_artifact_paths(args.output_dir, args.dataset, args.split)["predictions"]
    print(f"\nDone. Files written to {result['tasks_file'].parent}/")
    print(f"  Task list:    {result['tasks_file'].name}")
    print(f"  Agent prompt: {result['prompt_file'].name}")
    print(
        "\nNext: have the agent follow the prompt (it writes "
        f"{predictions.name}), then run:\n"
        f"  python -m vidlu_irap_gaim.tools.agent_classify evaluate "
        f"--dataset {args.dataset} --split {args.split} --predictions {predictions}"
    )


def _cmd_evaluate(args):
    evaluate_agent_predictions(
        predictions_file=args.predictions,
        dataset_name=args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        allow_invalid_predictions=args.allow_invalid_predictions,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Agent-based iRAP attribute classification (prepare/evaluate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Args common to both subcommands, declared once.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=["bih", "vietnam"], default="bih",
                        help="Dataset (default: bih)")
    common.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Dataset split (default: test)")
    common.add_argument("--output-dir", default="agent_run",
                        help="Shared run directory for the tasks, prompt, agent predictions, "
                             "and evaluation results (default: agent_run)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prepare = subparsers.add_parser(
        "prepare", parents=[common],
        help="Generate the agent task list and prompt")
    p_prepare.add_argument("--limit", type=int, default=None,
                           help="Maximum number of segments to include (default: all)")
    p_prepare.add_argument("--prompt-config", default=None,
                           help="Custom attribute_prompts.yaml path "
                                "(default: vlm/attribute_prompts.yaml)")
    p_prepare.set_defaults(func=_cmd_prepare)

    p_evaluate = subparsers.add_parser(
        "evaluate", parents=[common],
        help="Score the agent's predictions against ground truth")
    p_evaluate.add_argument("--predictions", required=True,
                            help="Agent predictions JSON (the file the agent wrote; "
                                 "the prepare step prints its path)")
    p_evaluate.add_argument("--allow-invalid-predictions", action="store_true", default=True,
                            help="Map unparseable values to class 0 (default: True)")
    p_evaluate.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
