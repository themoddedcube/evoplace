"""
Tests for the evolution evaluator wrapper and evolved program validation.

These tests verify that:
1. The initial seed program is valid and evaluatable
2. The evaluator wrapper correctly loads/rejects programs
3. Known-good and known-bad programs produce expected scores
"""

import math
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evolve.evaluator_wrapper import load_evolved_function, evaluate


VALID_GAMMA_CODE = textwrap.dedent("""\
    import math

    def gamma_schedule(iteration, total_iterations, overflow, hpwl_history):
        gamma_max = 8.0
        gamma_min = 0.5
        t = iteration / max(total_iterations - 1, 1)
        return gamma_max - (gamma_max - gamma_min) * t
""")

VALID_COSINE_CODE = textwrap.dedent("""\
    import math

    def gamma_schedule(iteration, total_iterations, overflow, hpwl_history):
        gamma_max = 8.0
        gamma_min = 0.5
        t = iteration / max(total_iterations - 1, 1)
        return gamma_min + (gamma_max - gamma_min) * 0.5 * (1 + math.cos(math.pi * t))
""")

OVERFLOW_ADAPTIVE_CODE = textwrap.dedent("""\
    def gamma_schedule(iteration, total_iterations, overflow, hpwl_history):
        gamma_max = 8.0
        gamma_min = 0.5
        t = iteration / max(total_iterations - 1, 1)
        base = gamma_max - (gamma_max - gamma_min) * t
        # Adapt: higher overflow -> lower gamma for better WL gradients
        return max(gamma_min, base * max(0.1, overflow))
""")

OUT_OF_RANGE_CODE = textwrap.dedent("""\
    def gamma_schedule(iteration, total_iterations, overflow, hpwl_history):
        return 100.0  # INVALID: out of [0.01, 20.0] range
""")

SYNTAX_ERROR_CODE = textwrap.dedent("""\
    def gamma_schedule(iteration, total_iterations, overflow, hpwl_history):
        return (  # unclosed parenthesis
""")

RUNTIME_ERROR_CODE = textwrap.dedent("""\
    def gamma_schedule(iteration, total_iterations, overflow, hpwl_history):
        return 1.0 / 0  # ZeroDivisionError
""")


def write_tmp_program(code: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
    f.write(code)
    f.flush()
    return Path(f.name)


class TestLoadEvolvedFunction:
    def test_loads_valid_function(self):
        path = write_tmp_program(VALID_GAMMA_CODE)
        fn = load_evolved_function(str(path), "gamma_schedule")
        assert callable(fn)

    def test_raises_on_wrong_function_name(self):
        path = write_tmp_program(VALID_GAMMA_CODE)
        with pytest.raises((AttributeError, ValueError)):
            load_evolved_function(str(path), "nonexistent_fn")

    def test_raises_on_syntax_error(self):
        path = write_tmp_program(SYNTAX_ERROR_CODE)
        with pytest.raises(SyntaxError):
            load_evolved_function(str(path), "gamma_schedule")

    def test_loaded_function_is_callable(self):
        path = write_tmp_program(VALID_GAMMA_CODE)
        fn = load_evolved_function(str(path), "gamma_schedule")
        val = fn(0, 1000, 0.9, [])
        assert isinstance(val, float)
        assert math.isfinite(val)


class TestEvaluatorWrapper:
    def test_baseline_returns_finite_score(self):
        path = write_tmp_program(VALID_GAMMA_CODE)
        result = evaluate(str(path), experiment="exp01_wl_smoothing")
        assert math.isfinite(result["score"])
        assert "metrics" in result

    def test_out_of_range_eliminated(self):
        path = write_tmp_program(OUT_OF_RANGE_CODE)
        result = evaluate(str(path), experiment="exp01_wl_smoothing")
        assert result["score"] == float("-inf")

    def test_runtime_error_eliminated(self):
        path = write_tmp_program(RUNTIME_ERROR_CODE)
        result = evaluate(str(path), experiment="exp01_wl_smoothing")
        assert result["score"] == float("-inf")

    def test_syntax_error_eliminated(self):
        path = write_tmp_program(SYNTAX_ERROR_CODE)
        result = evaluate(str(path), experiment="exp01_wl_smoothing")
        assert result["score"] == float("-inf")

    def test_cosine_schedule_evaluates(self):
        path = write_tmp_program(VALID_COSINE_CODE)
        result = evaluate(str(path), experiment="exp01_wl_smoothing")
        assert result["score"] != float("-inf")
        assert result["metrics"]["normalized_hpwl"] > 0

    def test_overflow_adaptive_evaluates(self):
        path = write_tmp_program(OVERFLOW_ADAPTIVE_CODE)
        result = evaluate(str(path), experiment="exp01_wl_smoothing")
        assert result["score"] != float("-inf")

    def test_metrics_dict_has_expected_keys(self):
        path = write_tmp_program(VALID_GAMMA_CODE)
        result = evaluate(str(path))
        for key in ["normalized_hpwl", "hpwl", "overflow", "runtime_s"]:
            assert key in result["metrics"], f"Missing key: {key}"

    def test_initial_program_file_evaluates(self):
        """The checked-in initial_program.py must be evaluatable."""
        initial = Path(__file__).parent.parent / "evolve" / "initial_program.py"
        assert initial.exists(), "initial_program.py missing"
        result = evaluate(str(initial), experiment="exp01_wl_smoothing")
        assert math.isfinite(result["score"])


class TestScheduleQualityRanking:
    """Verify that the stub evaluator rewards better schedules."""

    def score(self, code: str) -> float:
        path = write_tmp_program(code)
        result = evaluate(str(path), experiment="exp01_wl_smoothing")
        return result["score"]

    def test_monotone_schedule_beats_constant(self):
        constant_code = textwrap.dedent("""\
            def gamma_schedule(iteration, total_iterations, overflow, hpwl_history):
                return 4.0  # constant — no schedule at all
        """)
        linear_code = VALID_GAMMA_CODE
        s_const = self.score(constant_code)
        s_linear = self.score(linear_code)
        # Linear decay is monotone → should get stub reward; constant doesn't
        assert s_linear >= s_const
