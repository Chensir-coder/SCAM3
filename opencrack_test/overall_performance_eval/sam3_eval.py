import argparse
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
OPENCRACK_ROOT = CURRENT_DIR.parent
if str(OPENCRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENCRACK_ROOT))

from overall_performance_eval.Sam3_Eval import Sam3_Eval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM3 inference and evaluation on the OSRS crack dataset."
    )
    parser.add_argument(
        "--model-name",
        default="sam3",
        help="Name used to create output folders and result file names.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the SAM3 checkpoint, for example checkpoint/sam3/sam3.pt.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory of the OSRS crack dataset.",
    )
    parser.add_argument(
        "--json-path",
        required=True,
        help="Path to the dataset annotation JSON.",
    )
    parser.add_argument(
        "--exp-name",
        default="osrs_crack",
        help="Experiment name used in output file names.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where predictions, inference JSONs, and metric JSONs are saved.",
    )
    parser.add_argument(
        "--mode",
        choices=("prompt", "crack"),
        default="prompt",
        help=(
            "prompt: use each sample's prompts field; "
            "crack: use a fixed 'crack' prompt for every image."
        ),
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only create evaluation JSONs and compute metrics from existing predictions.",
    )
    parser.add_argument(
        "--skip-metric",
        action="store_true",
        help="Only run inference and save inference info JSON, without metric computation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_inference:
        evaluator = Sam3_Eval(
            model_name=args.model_name,
            model_path=args.model_path,
            work_dir=args.output_dir,
        )
        if args.mode == "prompt":
            evaluator.full_image_inference(
                exp_name=args.exp_name,
                dataset_root_path=args.dataset_root,
                osrs_crack_json_path=args.json_path,
                output_dir=args.output_dir,
            )
        else:
            evaluator.full_crack_inference(
                exp_name=args.exp_name,
                dataset_root_path=args.dataset_root,
                osrs_crack_json_path=args.json_path,
                output_dir=args.output_dir,
            )

    if not args.skip_metric:
        if args.mode == "prompt":
            Sam3_Eval.all_type_eval_json_create(
                model_name=args.model_name,
                exp_name=args.exp_name,
                dataset_root_path=args.dataset_root,
                osrs_crack_json_path=args.json_path,
                output_dir=args.output_dir,
            )
        else:
            inference_json = (
                Path(args.output_dir)
                / args.model_name
                / f"{args.exp_name}_inference_infos.json"
            )
            Sam3_Eval.compute_metric(inference_json)


if __name__ == "__main__":
    main()
