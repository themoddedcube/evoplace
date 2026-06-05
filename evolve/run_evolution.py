"""
Main entry point for running OpenEvolve on a placement experiment.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python evolve/run_evolution.py --experiment exp01_wl_smoothing
    python evolve/run_evolution.py --experiment exp02_density_schedule
    python evolve/run_evolution.py --experiment exp01_wl_smoothing --iterations 50  # quick test
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


EXPERIMENT_CONFIGS = {
    "exp01_wl_smoothing": {
        "initial_program": "evolve/initial_program.py",
        "config": "evolve/config.yaml",
        "function_name": "gamma_schedule",
        "description": "WA-WL smoothness (gamma) schedule evolution",
    },
    "exp02_density_schedule": {
        "initial_program": "evolve/initial_program_lambda.py",
        "config": "evolve/config_lambda.yaml",
        "function_name": "lambda_schedule",
        "description": "Density weight (lambda) schedule evolution",
    },
}


def run_with_openevolve(experiment: str, iterations: int, output_dir: Path):
    """Run evolution using the OpenEvolve library."""
    try:
        from openevolve import run_evolution
    except ImportError:
        logger.error(
            "OpenEvolve not installed. Run: pip install openevolve\n"
            "Or: pip install git+https://github.com/algorithmicsuperintelligence/openevolve.git"
        )
        sys.exit(1)

    cfg = EXPERIMENT_CONFIGS[experiment]
    initial_program = str(PROJECT_ROOT / cfg["initial_program"])
    config_path = str(PROJECT_ROOT / cfg["config"])

    # The evaluator wrapper — OpenEvolve calls evaluate(program_path)
    def evaluator(program_path: str):
        from evolve.evaluator_wrapper import evaluate
        return evaluate(program_path, experiment=experiment)

    logger.info(f"Starting evolution: {cfg['description']}")
    logger.info(f"Initial program: {initial_program}")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Output: {output_dir}")

    result = run_evolution(
        initial_program=initial_program,
        evaluator=evaluator,
        config=config_path,
        iterations=iterations,
        output_dir=str(output_dir),
    )

    return result


def detect_backend() -> str:
    """Pick an LLM backend: Claude Code CLI (no API key) > Anthropic API."""
    import shutil
    if shutil.which("claude"):
        return "claude-code-cli"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic-api"
    raise RuntimeError(
        "No LLM backend available: install the Claude Code CLI "
        "(https://claude.ai/code) or export ANTHROPIC_API_KEY."
    )


def _propose_with_llm(prompt: str, backend: str) -> str:
    """Send a mutation prompt to the chosen backend, return raw response text."""
    if backend == "claude-code-cli":
        import subprocess
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {proc.stderr[:500]}")
        return proc.stdout
    elif backend == "anthropic-api":
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    raise ValueError(f"Unknown backend: {backend}")


def run_autoresearch_loop(experiment: str, iterations: int, output_dir: Path,
                          backend: str = "auto"):
    """
    Autoresearch-style loop: simpler fallback if OpenEvolve is not installed.

    Asks an LLM (Claude Code CLI or Anthropic API) to iteratively improve
    the schedule function. Logs all results to results.tsv.
    """
    import importlib.util
    import hashlib

    if backend == "auto":
        backend = detect_backend()
    logger.info(f"LLM backend: {backend}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_tsv = output_dir / "results.tsv"

    cfg = EXPERIMENT_CONFIGS[experiment]
    current_program_path = PROJECT_ROOT / cfg["initial_program"]

    with open(current_program_path) as f:
        current_code = f.read()

    best_score = float("-inf")
    best_code = current_code

    # Write TSV header
    if not results_tsv.exists():
        with open(results_tsv, "w") as f:
            f.write("iteration\ttimestamp\tcode_hash\tnorm_hpwl\thpwl\toverflow\ttns_proxy\truntime_s\n")

    from evolve.evaluator_wrapper import evaluate

    for i in range(iterations):
        logger.info(f"\n=== Iteration {i+1}/{iterations} ===")

        # Evaluate current program
        tmp_path = output_dir / f"candidate_{i:04d}.py"
        with open(tmp_path, "w") as f:
            f.write(current_code)

        result = evaluate(str(tmp_path), experiment=experiment)
        score = result.get("score", float("-inf"))
        metrics = result.get("metrics", {})

        code_hash = hashlib.sha256(current_code.encode()).hexdigest()[:8]
        ts = int(time.time())

        with open(results_tsv, "a") as f:
            f.write(
                f"{i}\t{ts}\t{code_hash}\t"
                f"{metrics.get('normalized_hpwl', 'N/A')}\t"
                f"{metrics.get('hpwl', 'N/A')}\t"
                f"{metrics.get('overflow', 'N/A')}\t"
                f"{metrics.get('tns_proxy', 'N/A')}\t"
                f"{metrics.get('runtime_s', 'N/A')}\n"
            )

        logger.info(f"Score: {score:.4f} | norm_hpwl: {metrics.get('normalized_hpwl', '?'):.4f}")

        if score > best_score:
            best_score = score
            best_code = current_code
            with open(output_dir / "best_program.py", "w") as f:
                f.write(best_code)
            logger.info(f"New best! score={best_score:.4f}")
        else:
            # Revert to best
            current_code = best_code

        # Ask Claude to improve the function
        prompt = f"""You are an expert EDA algorithm researcher. Improve the following
placement schedule function to reduce HPWL (current normalized HPWL = {metrics.get('normalized_hpwl', '?'):.4f},
best so far = {-best_score:.4f}).

Current function:
```python
{current_code}
```

Rules:
- Preserve the function signature exactly
- Only modify the function body
- No new imports
- Return float in [0.01, 50.0]

Return ONLY the improved Python function, no explanation."""

        try:
            response = _propose_with_llm(prompt, backend)
        except Exception as e:
            logger.warning(f"LLM proposal failed ({e}); keeping current program")
            continue
        # Extract code from response
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response.strip()

        # Basic validation: must contain the function
        if cfg["function_name"] in code:
            current_code = code
        else:
            logger.warning("LLM response didn't contain expected function; keeping current")

    logger.info(f"\nEvolution complete. Best score: {best_score:.4f}")
    logger.info(f"Best program saved to {output_dir / 'best_program.py'}")
    return best_score, best_code


def main():
    parser = argparse.ArgumentParser(description="Run OpenEvolve on a placement experiment")
    parser.add_argument("--experiment", choices=list(EXPERIMENT_CONFIGS), required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--mode", choices=["openevolve", "autoresearch"], default="openevolve",
                        help="openevolve: full MAP-Elites evolution; autoresearch: simpler loop")
    parser.add_argument("--backend", choices=["auto", "claude-code-cli", "anthropic-api"],
                        default="auto",
                        help="LLM backend for autoresearch mode (auto: CLI if installed, else API)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = PROJECT_ROOT / "experiments" / args.experiment / "evolution_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Iterations: {args.iterations}")

    mode = args.mode
    if mode == "openevolve":
        try:
            import openevolve  # noqa: F401
        except ImportError:
            logger.warning("OpenEvolve not installed; falling back to autoresearch loop")
            mode = "autoresearch"

    if mode == "openevolve":
        result = run_with_openevolve(args.experiment, args.iterations, output_dir)
    else:
        result = run_autoresearch_loop(args.experiment, args.iterations, output_dir,
                                       backend=args.backend)

    logger.info("Done.")


if __name__ == "__main__":
    main()
