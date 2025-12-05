"""Tests for Windows-specific code paths and limitations."""

import sys
import unittest

from kalkulator_pkg.cli import _health_check
from kalkulator_pkg.worker import HAS_RESOURCE, _limit_resources


class TestWindowsResourceLimits(unittest.TestCase):
    """Test Windows-specific resource limit handling."""

    def test_resource_module_availability(self):
        """Test that resource module availability is correctly detected."""
        if sys.platform == "win32":
            # On Windows, resource module should not be available
            self.assertFalse(
                HAS_RESOURCE, "Resource module should not be available on Windows"
            )
        else:
            # On Unix, resource module should be available
            self.assertTrue(HAS_RESOURCE, "Resource module should be available on Unix")

    def test_limit_resources_on_windows(self):
        """Test that _limit_resources gracefully handles Windows (no resource module)."""
        if sys.platform == "win32":
            # Should not raise an error, just return silently
            try:
                _limit_resources()
                # Should succeed without raising
            except Exception as e:
                self.fail(
                    f"_limit_resources should handle Windows gracefully, but raised: {e}"
                )

    def test_health_check_windows_warning(self):
        """Test that health check includes Windows resource limit warning."""
        if sys.platform == "win32":
            # Run health check and capture output
            import io
            from contextlib import redirect_stdout

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = _health_check()  # noqa: F841

            output_str = output.getvalue()
            # Should mention Windows or resource limits or DEPLOYMENT
            output_lower = output_str.lower()
            has_warning = (
                "windows" in output_lower
                or "resource" in output_lower
                or "deployment" in output_lower
                or "containerization" in output_lower
            )
            self.assertTrue(
                has_warning,
                f"Health check should mention Windows limitations. Output: {output_str}",
            )

    def test_worker_evaluation_on_windows(self):
        """Test that worker evaluation works on Windows without resource limits."""
        from kalkulator_pkg.worker import evaluate_safely

        # Should work even without resource module
        result = evaluate_safely("2 + 2")
        self.assertTrue(result.get("ok"), "Evaluation should work on Windows")
        self.assertEqual(result.get("result"), "4")


class TestWindowsCompatibility(unittest.TestCase):
    """Test general Windows compatibility."""

    def test_unicode_output_on_windows(self):
        """Test that Unicode output is handled gracefully on Windows."""
        import io
        from contextlib import redirect_stdout

        from kalkulator_pkg.cli import print_result_pretty

        # Test with Unicode characters that might cause encoding issues
        result = {
            "ok": True,
            "type": "pell",
            "solution": "x = (3 + 2√5)^n + (3 - 2√5)^n / 2",  # Contains Unicode √
        }

        output = io.StringIO()
        try:
            with redirect_stdout(output):
                print_result_pretty(result)
            # Should not raise UnicodeEncodeError
            output_str = output.getvalue()
            self.assertTrue(len(output_str) > 0, "Output should be generated")
        except UnicodeEncodeError as e:
            self.fail(f"Unicode encoding error should be handled: {e}")


if __name__ == "__main__":
    unittest.main()
