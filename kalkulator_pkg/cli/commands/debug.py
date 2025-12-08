from ..context import ReplContext
import logging

def handle_debug_command(ctx: ReplContext, cmd: str) -> None:
    """Handle the 'debug' command."""
    parts = str(cmd).split()
    if len(parts) > 1:
        mode = parts[1].lower()
        if mode in ("on", "true", "enabled"):
            ctx.timing_enabled = True
            ctx.cache_hits_enabled = True
            try:
                logging.getLogger().setLevel(logging.DEBUG)
                print("Debug mode enabled (timing + logging + cache hits).")
            except Exception as e:
                print(f"Debug mode enabled (timing + cache hits). details: {e}")
        elif mode in ("off", "false", "disabled"):
            ctx.timing_enabled = False
            ctx.cache_hits_enabled = False
            try:
                logging.getLogger().setLevel(logging.WARNING)
                print("Debug mode disabled.")
            except Exception:
                print("Debug mode disabled.")
        else:
            print("Usage: debug <on|off>")
    else:
            print("Usage: debug <on|off>")
