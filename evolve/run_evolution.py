"""
Main entry point for running OpenEvolve on a placement experiment.

Usage (no API key needed — uses Claude Code CLI):
    python evolve/run_evolution.py --experiment exp01_wl_smoothing
    python evolve/run_evolution.py --experiment exp01_wl_smoothing --iterations 50

Backend selection (auto-detected, or set explicitly):
    --backend claude-code-cli   # default: uses `claude -p` subprocess (your CC subscription)
    --backend anthropic-api     # needs ANTHROPIC_API_KEY env var
    --backend gemini            # needs GEMINI_API_KEY env var
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
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


# ---------------------------------------------------------------------------
# LLM backend abstraction
# ---------------------------------------------------------------------------

def _detect_backend() -> str:
    """Auto-detect which LLM backend is available, in priority order."""
    if subprocess.run(["which", "claude"], capture_output=True).returncode == 0:
        return "claude-code-cli"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic-api"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    raise RuntimeError(
        "No LLM backend available. Options:\n"
        "  1. Install Claude Code CLI (already done if you're reading this)\n"
        "  2. export ANTHROPIC_API_KEY=sk-ant-...\n"
        "  3. export GEMINI_API_KEY=... (free at aistudio.google.com)"
    )


def call_llm(prompt: str, system: str, backend: str, model: str = None) -> str:
    """
    Call an LLM with a prompt and return the text response.

    Supports three backends:
    - claude-code-cli: subprocess `claude -p` (uses Claude Code subscription, no key needed)
    - anthropic-api:   direct Anthropic SDK call (needs ANTHROPIC_API_KEY)
    - gemini:          Google Generative AI SDK (needs GEMINI_API_KEY, free tier available)
    """
    if backend == "claude-code-cli":
        return _call_claude_code_cli(prompt, system)
    elif backend == "anthropic-api":
        return _call_anthropic_api(prompt, system, model or "claude-sonnet-4-6")
    elif backend == "gemini":
        return _call_gemini(prompt, system, model or "gemini-2.0-flash")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _call_claude_code_cli(prompt: str, system: str) -> str:
    """Call Claude Code CLI in print mode. No API key required."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    result = subprocess.run(
        ["claude", "-p", full_prompt],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {result.stderr[:500]}")
    return result.stdout.strip()


def _call_anthropic_api(prompt: str, system: str, model: str) -> str:
    """Call Anthropic API directly. Requires ANTHROPIC_API_KEY."""
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_gemini(prompt: str, system: str, model: str) -> str:
    """Call Google Gemini API. Free tier at aistudio.google.com."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Run: pip install google-generativeai")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(
        model_name=model,
        system_instruction=system,
    )
    response = m.generate_content(prompt)
    return response.text


def _extract_function(response: str, function_name: str) -> str | None:
    """Pull the Python function out of an LLM response."""
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0].strip()
    elif "```" in response:
        code = response.split("```")[1].split("```")[0].strip()
    else:
        code = response.strip()
    return code if function_name in code else None


# ---------------------------------------------------------------------------
# OpenEvolve (MAP-Elites) path
# ---------------------------------------------------------------------------

def run_with_openevolve(experiment: str, iterations: int, output_dir: Path, backend: str):
    """Run evolution using the OpenEvolve library."""
    try:
        from openevolve import run_evolution
    except ImportError:
        logger.error(
            "OpenEvolve not installed. Run: pip install openevolve\n"
            "Or fall back with: --mode autoresearch"
        )
        sys.exit(1)

    cfg = EXPERIMENT_CONFIGS[experiment]
    initial_program = str(PROJECT_ROOT / cfg["initial_program"])
    config_path = str(PROJECT_ROOT / cfg["config"])

    def evaluator(program_path: str):
        from evolve.evaluator_wrapper import evaluate
        return evaluate(program_path, experiment=experiment)

    logger.info(f"Starting evolution: {cfg['description']}")
    logger.info(f"Backend: {backend}")

    result = run_evolution(
        initial_program=initial_program,
        evaluator=evaluator,
        config=config_path,
        iterations=iterations,
        output_dir=str(output_dir),
    )
    return result


# ---------------------------------------------------------------------------
# Autoresearch loop (simpler, no OpenEvolve dependency)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert EDA (Electronic Design Automation) algorithm researcher
specializing in VLSI placement. Your task is to evolve a schedule function
to minimize HPWL (half-perimeter wirelength) in differentiable global placement.

Context for gamma (γ) schedule:
- gamma controls WA-WL approximation smoothness in DREAMPlace
- High γ (8.0): smooth gradients, inaccurate HPWL approximation
- Low γ (0.5): accurate HPWL, noisy gradients
- overflow: fraction of bins over-density (1.0 = all bins full)
- Good schedule: high γ early (cells clustered) → low γ late (fine-tuning)
- Known techniques: linear/exponential decay, overflow-adaptive, cosine annealing, piecewise

Rules (MUST follow):
- Preserve the exact function signature
- Only modify the function body
- No new imports, no file I/O, no external calls
- Return a float in [0.01, 20.0]
- Return ONLY the Python function, no explanation"""


def run_autoresearch_loop(experiment: str, iterations: int, output_dir: Path, backend: str):
    """
    Simple hill-climbing evolution loop. No OpenEvolve dependency.
    Calls the configured LLM backend to generate each mutation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_tsv = output_dir / "results.tsv"

    cfg = EXPERIMENT_CONFIGS[experiment]
    current_program_path = PROJECT_ROOT / cfg["initial_program"]
    function_name = cfg["function_name"]

    with open(current_program_path) as f:
        current_code = f.read()

    best_score = float("-inf")
    best_code = current_code

    if not results_tsv.exists():
        with open(results_tsv, "w") as f:
            f.write("iteration\ttimestamp\tcode_hash\tnorm_hpwl\thpwl\toverflow\ttns_proxy\truntime_s\n")

    from evolve.evaluator_wrapper import evaluate

    logger.info(f"Starting autoresearch loop: {cfg['description']}")
    logger.info(f"LLM backend: {backend}")
    logger.info(f"Iterations: {iterations}")

    for i in range(iterations):
        logger.info(f"\n=== Iteration {i+1}/{iterations} ===")

        tmp_path = output_dir / f"candidate_{i:04d}.py"
        with open(tmp_path, "w") as f:
            f.write(current_code)

        result = evaluate(str(tmp_path), experiment=experiment)
        score = result.get("score", float("-inf"))
        metrics = result.get("metrics", {})

        code_hash = hashlib.sha256(current_code.encode()).hexdigest()[:8]
        norm_hpwl = metrics.get("normalized_hpwl", float("inf"))

        with open(results_tsv, "a") as f:
            f.write(
                f"{i}\t{int(time.time())}\t{code_hash}\t"
                f"{metrics.get('normalized_hpwl', 'N/A')}\t"
                f"{metrics.get('hpwl', 'N/A')}\t"
                f"{metrics.get('overflow', 'N/A')}\t"
                f"{metrics.get('tns_proxy', 'N/A')}\t"
                f"{metrics.get('runtime_s', 'N/A')}\n"
            )

        logger.info(f"Score: {score:.4f} | norm_hpwl: {norm_hpwl:.4f}")

        if score > best_score:
            best_score = score
            best_code = current_code
            with open(output_dir / "best_program.py", "w") as f:
                f.write(best_code)
            logger.info(f"New best! score={best_score:.4f}")
        else:
            current_code = best_code  # revert

        # Strip docstring and comments from code to keep prompt short
        import re as _re
        compact_code = _re.sub(r'""".*?"""', '""" ... """', current_code, flags=_re.DOTALL)
        compact_code = _re.sub(r'#.*\n', '\n', compact_code)
        compact_code = _re.sub(r'\n{3,}', '\n\n', compact_code).strip()

        user_prompt = (
            f"Current function:\n```python\n{compact_code}\n```\n\n"
            f"Current performance: normalized_hpwl = {norm_hpwl:.4f} "
            f"(best so far: {-best_score:.4f})\n\n"
            f"Generate an improved {function_name}. Return ONLY the Python function."
        )

        try:
            response = call_llm(user_prompt, SYSTEM_PROMPT, backend)
            new_code = _extract_function(response, function_name)
            if new_code:
                current_code = new_code
                logger.info("LLM produced valid mutation")
            else:
                logger.warning("LLM response didn't contain the expected function; keeping current")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")

    logger.info(f"\nEvolution complete. Best score: {best_score:.4f}")
    logger.info(f"Best program: {output_dir / 'best_program.py'}")
    return best_score, best_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run schedule evolution for EvoPlace")
    parser.add_argument("--experiment", choices=list(EXPERIMENT_CONFIGS), required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--mode", choices=["openevolve", "autoresearch"], default="autoresearch",
        help="autoresearch: simple hill-climb loop (default); openevolve: MAP-Elites"
    )
    parser.add_argument(
        "--backend", choices=["claude-code-cli", "anthropic-api", "gemini"], default=None,
        help="LLM backend (auto-detected if not set)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override default output directory (default: experiments/<experiment>/evolution_runs)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    backend = args.backend or _detect_backend()
    logger.info(f"LLM backend: {backend}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "experiments" / args.experiment / "evolution_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Iterations: {args.iterations}")

    if args.mode == "openevolve":
        result = run_with_openevolve(args.experiment, args.iterations, output_dir, backend)
    else:
        result = run_autoresearch_loop(args.experiment, args.iterations, output_dir, backend)

    logger.info("Done.")


if __name__ == "__main__":
    main()
