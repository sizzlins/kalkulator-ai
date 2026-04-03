"""
AI Utility functions for Kalkulator.
Handles integration with external AI tools like Kimi CLI.
"""
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class KimiWrapper:
    """Wrapper for the Kimi AI CLI tool."""
    
    def __init__(self):
        self._executable = self._find_executable()
        
    def _find_executable(self) -> Optional[str]:
        """Locate the kimi executable."""
        # Check standard PATH
        exe_path = shutil.which("kimi")
        if exe_path:
            return exe_path
            
        # Check ~/.local/bin (where uv installs it on Windows sometimes)
        user_profile = os.environ.get("USERPROFILE", "")
        if user_profile:
            local_bin = Path(user_profile) / ".local" / "bin" / "kimi.exe"
            if local_bin.exists():
                return str(local_bin)
                
            # Check ~/.kimi/bin (standard install)
            kimi_bin = Path(user_profile) / ".kimi" / "bin" / "kimi.exe"
            if kimi_bin.exists():
                return str(kimi_bin)
                
        # Check LocalAppData
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            kimi_exe = Path(local_app_data) / "kimi" / "kimi.exe"
            if kimi_exe.exists():
                return str(kimi_exe)
                
        return None

    def is_available(self) -> bool:
        """Check if Kimi CLI is available."""
        return self._executable is not None

    def query(self, prompt: str) -> str:
        """Send a query to Kimi CLI and return the response."""
        if not self._executable:
            return "Error: Kimi CLI not found. Please install it first."
            
        try:
            # Run kimi --quiet with prompt in stdin
            # --quiet is alias for --print --output-format text --final-message-only
            process = subprocess.Popen(
                [self._executable, "--quiet"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            stdout, stderr = process.communicate(input=prompt)
            
            if process.returncode != 0:
                # Check if it was just empty output or actual error
                if stderr:
                    return f"Error running Kimi: {stderr}"
                # Sometimes return code is non-zero but output exists? 
                # But usually error implies failure.
                return f"Error running Kimi (Exit {process.returncode}): {stderr}"
            
            return stdout.strip()

        except Exception as e:
            logger.error(f"Failed to query Kimi: {e}")
            return f"Error executing Kimi: {str(e)}"

# Singleton instance
_kimi_instance = None

def get_kimi() -> KimiWrapper:
    global _kimi_instance
    if _kimi_instance is None:
        _kimi_instance = KimiWrapper()
    return _kimi_instance
