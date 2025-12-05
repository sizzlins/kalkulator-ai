# Kalkulator Project Organization

This document describes the organization and structure of the Kalkulator project.

## Directory Structure

```
Kalkulator/
├── kalkulator.py              # Main entry point
├── kalkulator_pkg/            # Main package directory
│   ├── __init__.py
│   ├── __main__.py            # Module entry point (python -m kalkulator_pkg)
│   ├── api.py                 # Public API
│   ├── cache_manager.py       # Persistent cache management
│   ├── calculus.py            # Differentiation, integration
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Configuration constants
│   ├── logging_config.py      # Logging setup
│   ├── parser.py              # Expression parsing
│   ├── plotting.py            # Plotting functionality
│   ├── solver.py              # Equation/inequality solving
│   ├── types.py               # Type definitions
│   └── worker.py              # Worker processes
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_api_typed_returns.py
│   ├── test_calculus.py
│   ├── test_comprehensive.py
│   ├── test_error_codes.py
│   ├── test_failure_modes.py
│   ├── test_fuzzing.py
│   ├── test_integration.py
│   ├── test_integration_cli.py
│   ├── test_parser.py
│   ├── test_performance.py
│   ├── test_solver.py
│   ├── test_solver_handlers.py
│   └── test_windows_specific.py
│
├── docs/                      # Documentation
│   ├── archive/               # Historical/archived documents
│   │   ├── STATUS_ARCHIVE.md  # Archive index
│   │   └── ...                # Various archived status files
│   ├── status/                # Status tracking documents
│   │   ├── CODE_REVIEW_TODO.md
│   │   ├── TODO_*.md          # Various TODO tracking files
│   │   └── PROGRESS_*.md      # Progress tracking files
│   ├── PROJECT_STRUCTURE.md   # Project structure details
│   ├── WORKER_SUBPROCESS_FALLBACK.md
│   └── PROJECT_ORGANIZATION.md (this file)
│
├── build/                     # Build artifacts (gitignored)
│   ├── coverage.xml
│   └── kalkulator.spec
│
├── .coverage_html/            # Coverage HTML reports (gitignored)
│
├── README.md                  # Main project README
├── LICENSE                    # MIT License
├── ONBOARDING.md              # Onboarding guide
├── ARCHITECTURE.md            # Architecture documentation
├── API.md                     # API documentation
├── SECURITY.md                # Security documentation
├── CONTRIBUTING.md            # Contribution guidelines
├── DEPLOYMENT.md              # Deployment guide
├── ENVIRONMENT_VARIABLES.md   # Environment variables
├── CHANGELOG.md               # Change log
├── RELEASE_NOTES.md           # Release notes
├── CODE_REVIEW_TODO.md        # Active code review TODO (symlink to docs/status/)
├── pyproject.toml             # Python project configuration
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── qodana.yaml                # Qodana configuration
└── run_formatting_tools.ps1   # Formatting script
```

## File Categories

### Core Application Files
- `kalkulator.py` - Main entry point script
- `kalkulator_pkg/` - Main Python package

### Documentation (Root)
- `README.md` - Main project documentation
- `LICENSE` - MIT License
- `ARCHITECTURE.md` - System architecture
- `API.md` - Public API reference
- `SECURITY.md` - Security practices
- `CONTRIBUTING.md` - Contribution guide
- `DEPLOYMENT.md` - Deployment instructions
- `ENVIRONMENT_VARIABLES.md` - Environment variable docs
- `CHANGELOG.md` - Version history
- `RELEASE_NOTES.md` - Release notes
- `ONBOARDING.md` - New developer guide

### Documentation (docs/)
- `docs/archive/` - Historical/archived status files
- `docs/status/` - Active status tracking files (TODO lists, progress reports)
- `docs/PROJECT_STRUCTURE.md` - Detailed project structure
- `docs/WORKER_SUBPROCESS_FALLBACK.md` - Worker fallback documentation

### Build and Artifacts (gitignored)
- `build/` - Build artifacts (spec files, coverage reports)
- `.coverage_html/` - HTML coverage reports
- `__pycache__/` - Python bytecode cache
- `htmlcov/` - Coverage HTML (deprecated, use .coverage_html/)

### Configuration Files
- `pyproject.toml` - Python project metadata and build config
- `requirements.txt` - Production dependencies
- `requirements-dev.txt` - Development dependencies
- `qodana.yaml` - Code quality tool config
- `.gitignore` - Git ignore patterns

### Scripts
- `run_formatting_tools.ps1` - Code formatting automation

## Organization Principles

1. **Root Directory**: Contains only essential files (entry point, main docs, config)
2. **kalkulator_pkg/**: All source code organized by functionality
3. **tests/**: All test files in dedicated directory
4. **docs/**: All documentation organized by purpose:
   - Root-level docs: User-facing and essential
   - `docs/`: Internal, archived, and status tracking
5. **build/**: All build artifacts consolidated
6. **Status Files**: Consolidated in `docs/status/` for easy tracking

## Maintenance

- Status tracking files (`*TODO*`, `*PROGRESS*`, `*REMAINING*`) should be kept in `docs/status/`
- Build artifacts should go in `build/`
- Coverage reports should go in `.coverage_html/`
- Archived documents should go in `docs/archive/`

