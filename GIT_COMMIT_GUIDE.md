# Git Commit Guide for Kalkulator

## Current Status

After cleanup, here's what should be committed:

### ✅ Files to Add

**Source Code:**
```bash
git add kalkulator.py
git add kalkulator_pkg/
git add tests/
```

**Documentation:**
```bash
git add *.md
git add docs/
```

**Configuration:**
```bash
git add pyproject.toml
git add requirements.txt
git add requirements-dev.txt
git add .gitignore
git add .github/
git add .pre-commit-config.yaml
git add qodana.yaml
```

**Scripts:**
```bash
git add run_formatting_tools.ps1
```

**License:**
```bash
git add LICENSE
```

### ❌ Files Removed from Tracking

- `.idea/` - Removed (IDE settings, now ignored)
- `__pycache__/` - Removed (Python cache, now ignored)
- Old cleanup files - Properly deleted

### Recommended Commit Commands

**Option 1: Add all safe files**
```bash
# Add all new and modified files
git add -A

# Verify what will be committed
git status

# Commit
git commit -m "feat: Enhanced Kalkulator with persistent cache, improved error handling, and expanded CLI commands

- Added persistent cache system with sub-expression caching
- Implemented cache save/load commands (savecache, loadcache)
- Added timing command for performance monitoring
- Enhanced error handling with better messages and hints
- Improved TokenError and syntax error detection
- Organized project structure (moved files to docs/status/)
- Added module entry point (python -m kalkulator_pkg)
- Updated .gitignore for cache files and build artifacts"
```

**Option 2: Stage specific categories**
```bash
# Source code
git add kalkulator.py kalkulator_pkg/ tests/

# Documentation
git add *.md docs/

# Configuration
git add pyproject.toml requirements*.txt .gitignore .github/ .pre-commit-config.yaml qodana.yaml LICENSE

# Scripts
git add run_formatting_tools.ps1

# Commit
git commit -m "feat: Enhanced Kalkulator with persistent cache and improved error handling"
```

### Verification Before Commit

Check that no unwanted files are included:
```bash
git status
```

Ensure you see:
- ✅ Source code files
- ✅ Documentation files
- ✅ Configuration files
- ❌ NO `__pycache__/` files
- ❌ NO `.idea/` files
- ❌ NO `venv/` or `.venv/` directories
- ❌ NO cache files

### After Committing

Verify the commit:
```bash
git show --stat
```

This should show only source code, documentation, and config files.

