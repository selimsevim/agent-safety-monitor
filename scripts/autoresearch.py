"""
autoresearch.py

Adapted from Karpathy's autoresearch pattern.
Runs overnight hyperparameter search for the safety classifier.
Agent varies LoRA rank, learning rate, and classification threshold.
Metric: False Negative Rate on validation set. Lower is better.
Keeps any run that improves FNR without pushing FPR above MAX_FPR.

Usage:
    python scripts/autoresearch.py \
        --train data/processed/train_augmented.jsonl \
        --val data/processed/val.jsonl \
        --base-model microsoft/Phi-3-mini-4k-instruct \
        --output models/adapters \
        --budget 8   # hours
"""

import json
import time
import argparse
import subprocess
import random
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product

# -----------------------------------------------------------------------
# Search space
# These are the only things the loop varies.
# Everything else stays fixed.
# -----------------------------------------------------------------------
LORA_RANKS = [8, 16, 32]
LEARNING_RATES = [1e-4, 2e-4, 5e-4]
EPOCHS_OPTIONS = [2, 3]
THRESHOLDS = [0.3, 0.5, 0.7]

# Safety constraint: never accept a run that pushes FPR above this.
MAX_FPR = 0.30

# Cost function: same as the report.
# Lower is better.
def cost(fnr: float, fpr: float, n: int) -> float:
    fn = round(fnr * n)
    fp = round(fpr * n)
    return fn * 10 + fp * 1


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def log(msg: str, log_path: Path):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(log_path, 'a') as f:
        f.write(line + "\n")


def run_training(
    base_model: str,
    train_path: str,
    val_path: str,
    output_dir: str,
    lora_rank: int,
    lr: float,
    epochs: int
) -> bool:
    """Run train.py as subprocess. Returns True if completed successfully."""
    cmd = [
        "python", "scripts/train.py",
        "--model", base_model,
        "--train", train_path,
        "--val", val_path,
        "--output", output_dir,
        "--lora-rank", str(lora_rank),
        "--epochs", str(epochs),
        "--lr", str(lr),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def run_evaluation(
    base_model: str,
    adapter_path: str,
    test_path: str,
    output_path: str,
    threshold: float
) -> dict:
    """Run evaluate.py and return metrics dict."""
    cmd = [
        "python", "scripts/evaluate.py",
        "--test", test_path,
        "--adapter", adapter_path,
        "--base-model", base_model,
        "--output", output_path,
        "--threshold", str(threshold),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return load_metrics(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/processed/train_augmented.jsonl')
    parser.add_argument('--val', default='data/processed/val.jsonl')
    parser.add_argument('--base-model', default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--output', default='models/adapters')
    parser.add_argument('--budget', type=float, default=8.0,
                        help='Time budget in hours')
    parser.add_argument('--log', default='results/autoresearch_log.jsonl')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    log_path = Path('results/autoresearch_run.log')
    log_path.parent.mkdir(exist_ok=True)

    deadline = datetime.now() + timedelta(hours=args.budget)

    log(f"Starting autoresearch. Budget: {args.budget}h. Deadline: {deadline.strftime('%H:%M:%S')}", log_path)

    # Build search grid and shuffle it
    # This means we explore randomly rather than sequentially,
    # so even a partial run gives useful coverage.
    grid = list(product(LORA_RANKS, LEARNING_RATES, EPOCHS_OPTIONS, THRESHOLDS))
    random.shuffle(grid)

    log(f"Search space: {len(grid)} configurations", log_path)

    # Track best run
    best_config = None
    best_fnr = float('inf')
    best_cost = float('inf')
    best_adapter_path = None

    all_results = []
    run_index = 0

    for lora_rank, lr, epochs, threshold in grid:
        if datetime.now() >= deadline:
            log("Time budget exhausted. Stopping.", log_path)
            break

        run_index += 1
        run_id = f"run_{timestamp()}_{run_index:03d}"
        adapter_path = str(output_dir / run_id)
        eval_output = str(results_dir / f"{run_id}_eval.json")

        config = {
            'run_id': run_id,
            'lora_rank': lora_rank,
            'lr': lr,
            'epochs': epochs,
            'threshold': threshold,
        }

        log(f"Run {run_index}: rank={lora_rank} lr={lr} epochs={epochs} threshold={threshold}", log_path)

        # Train
        t0 = time.time()
        success = run_training(
            base_model=args.base_model,
            train_path=args.train,
            val_path=args.val,
            output_dir=adapter_path,
            lora_rank=lora_rank,
            lr=lr,
            epochs=epochs
        )
        train_time = time.time() - t0

        if not success:
            log(f"  Training failed. Skipping.", log_path)
            config['status'] = 'train_failed'
            all_results.append(config)
            continue

        # Evaluate on validation set
        metrics = run_evaluation(
            base_model=args.base_model,
            adapter_path=adapter_path,
            test_path=args.val,  # use val for search, test is held out
            output_path=eval_output,
            threshold=threshold
        )

        if metrics is None:
            log(f"  Evaluation failed. Skipping.", log_path)
            config['status'] = 'eval_failed'
            all_results.append(config)
            shutil.rmtree(adapter_path, ignore_errors=True)
            continue

        fine_tuned = metrics.get('fine_tuned', {})
        run_fnr = fine_tuned.get('fnr', 1.0)
        run_fpr = fine_tuned.get('fpr', 1.0)
        n = metrics.get('n_test', 1)
        run_cost = cost(run_fnr, run_fpr, n)

        log(f"  FNR={run_fnr:.4f} FPR={run_fpr:.4f} cost={run_cost} time={train_time/60:.1f}min", log_path)

        config['status'] = 'completed'
        config['fnr'] = run_fnr
        config['fpr'] = run_fpr
        config['cost'] = run_cost
        config['train_time_minutes'] = round(train_time / 60, 1)
        all_results.append(config)

        # Keep or discard
        # Accept if: FNR improves AND FPR stays under MAX_FPR
        if run_fpr <= MAX_FPR and run_fnr < best_fnr:
            log(f"  NEW BEST: FNR {best_fnr:.4f} -> {run_fnr:.4f}", log_path)

            # Remove previous best adapter to save disk
            if best_adapter_path and Path(best_adapter_path).exists():
                shutil.rmtree(best_adapter_path, ignore_errors=True)

            best_fnr = run_fnr
            best_cost = run_cost
            best_config = config
            best_adapter_path = adapter_path
        else:
            reason = "FPR too high" if run_fpr > MAX_FPR else "FNR did not improve"
            log(f"  Discarded ({reason})", log_path)
            shutil.rmtree(adapter_path, ignore_errors=True)

        # Write running log after every experiment
        with open(args.log, 'w') as f:
            json.dump({
                'best': best_config,
                'best_adapter': best_adapter_path,
                'all_runs': all_results
            }, f, indent=2)

    # Final summary
    log("=" * 60, log_path)
    log(f"Autoresearch complete. {run_index} runs in {args.budget}h.", log_path)

    if best_config:
        log(f"Best config: {best_config}", log_path)
        log(f"Best FNR: {best_fnr:.4f}", log_path)
        log(f"Best adapter: {best_adapter_path}", log_path)

        # Copy best adapter to stable path
        stable_path = output_dir / "best"
        if stable_path.exists():
            shutil.rmtree(stable_path)
        shutil.copytree(best_adapter_path, stable_path)
        log(f"Best adapter copied to {stable_path}", log_path)
    else:
        log("No successful runs found.", log_path)

    # Final results file
    final_results = {
        'summary': {
            'total_runs': run_index,
            'successful_runs': sum(1 for r in all_results if r.get('status') == 'completed'),
            'budget_hours': args.budget,
            'best_fnr': best_fnr if best_config else None,
            'best_config': best_config,
            'best_adapter_path': str(output_dir / "best") if best_config else None,
        },
        'all_runs': all_results
    }

    with open('results/autoresearch_final.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\nResults saved to results/autoresearch_final.json")
    print(f"Best adapter at: {output_dir / 'best'}")


if __name__ == '__main__':
    main()
