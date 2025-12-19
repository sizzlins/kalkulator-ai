import unittest
from io import StringIO
from unittest.mock import patch

from kalkulator_pkg.cli import repl_loop


class TestDebugCommand(unittest.TestCase):
    def test_debug_on_off(self):
        commands = ["debug on", "y=2", "debug off", "quit"]

        with patch("builtins.input", side_effect=commands), patch(
            "sys.stdout", new_callable=StringIO
        ) as mock_stdout, patch("logging.getLogger"):

            try:
                repl_loop()
            except SystemExit:
                pass

            output = mock_stdout.getvalue()
            print(output)

            self.assertIn("Debug mode enabled", output)
            self.assertIn("Debug mode disabled", output)

            # Verify logging level set
            # mock_logger.return_value.setLevel.assert_called()


if __name__ == "__main__":
    unittest.main()
