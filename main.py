import argparse
import time

from src import compare_models, explore, final_evaluation, ingest, train_baseline
from src.config import load_config

# Pipeline steps, in the order they run
STEPS = {
    "ingest": ingest.run,
    "explore": explore.run,
    "baseline": train_baseline.run,
    "compare": compare_models.run,
    "final": final_evaluation.run,
}


def main():
    parser = argparse.ArgumentParser(description="Run the failure prediction pipeline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument(
        "--steps", nargs="+", choices=list(STEPS), default=list(STEPS),
        help="Which steps to run (default: all). They always run in pipeline order.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Pipeline: {config['name']}  |  steps: {', '.join(args.steps)}")

    for step, run_step in STEPS.items():
        if step not in args.steps:
            continue
        print(f"\n{'=' * 20} {step.upper()} {'=' * 20}")
        start = time.time()
        run_step(config)
        print(f"[{step} done in {time.time() - start:.0f}s]")


if __name__ == "__main__":
    main()