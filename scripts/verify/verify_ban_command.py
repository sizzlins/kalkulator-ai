
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from kalkulator_pkg.cli.context import ReplContext
from kalkulator_pkg.cli.repl_commands import handle_command, _handle_ban_command, _handle_unban_command

def test_ban_logic():
    print("Testing Ban Command Logic...")
    ctx = ReplContext()
    variables = {}

    # 1. Test BAN
    print("\n1. Testing 'ban sin, cos'")
    handle_command("ban sin, cos", ctx, variables)
    
    assert "sin" in ctx.banned_operators, "sin should be banned"
    assert "cos" in ctx.banned_operators, "cos should be banned"
    assert len(ctx.banned_operators) == 2
    print("MATCH: Bans applied correctly.")

    # 2. Test Partial UNBAN
    print("\n2. Testing 'unban sin'")
    handle_command("unban sin", ctx, variables)
    
    assert "sin" not in ctx.banned_operators, "sin should be unbanned"
    assert "cos" in ctx.banned_operators, "cos should remain banned"
    assert len(ctx.banned_operators) == 1
    print("MATCH: Partial unban successful.")

    # 3. Test Cumulative BAN
    print("\n3. Testing 'ban tan'")
    handle_command("ban tan", ctx, variables)
    
    assert "tan" in ctx.banned_operators
    assert "cos" in ctx.banned_operators
    assert len(ctx.banned_operators) == 2
    print("MATCH: Cumulative ban successful.")

    # 4. Test UNBAN ALL
    print("\n4. Testing 'unban all'")
    handle_command("unban all", ctx, variables)
    
    assert len(ctx.banned_operators) == 0, "All bans should be cleared"
    print("MATCH: Unban all successful.")

    print("\nSUCCESS: All ban/unban logic verified locally.")

if __name__ == "__main__":
    test_ban_logic()
