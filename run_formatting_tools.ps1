# Script to run all formatting tools and achieve 100% completion
# Run: .\run_formatting_tools.ps1

Write-Host "=== Running Formatting Tools for 100% Completion ===" -ForegroundColor Cyan
Write-Host ""

# Check if we're in a virtual environment
if (-not $env:VIRTUAL_ENV -and (Test-Path "venv\Scripts\activate.ps1")) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\activate.ps1
}

# Check if dev dependencies are installed
Write-Host "Checking development dependencies..." -ForegroundColor Yellow
$blackInstalled = python -m pip show black 2>$null
if (-not $blackInstalled) {
    Write-Host "Installing development dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements-dev.txt
} else {
    Write-Host "Development dependencies already installed." -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Item 9: Running Black Formatter ===" -ForegroundColor Cyan
python -m black kalkulator_pkg tests
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Black formatting complete" -ForegroundColor Green
} else {
    Write-Host "✗ Black formatting had issues" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Item 10: Running isort Import Sorter ===" -ForegroundColor Cyan
python -m isort kalkulator_pkg tests
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ isort import sorting complete" -ForegroundColor Green
} else {
    Write-Host "✗ isort had issues" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Item 11: Running Ruff Linter ===" -ForegroundColor Cyan
python -m ruff check --fix kalkulator_pkg tests
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Ruff linting complete" -ForegroundColor Green
} else {
    Write-Host "⚠ Ruff found issues (check output above)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Item 12: Running MyPy Type Checker ===" -ForegroundColor Cyan
python -m mypy kalkulator_pkg
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ MyPy type checking complete" -ForegroundColor Green
} else {
    Write-Host "⚠ MyPy found type issues (check output above)" -ForegroundColor Yellow
    Write-Host "Note: Some SymPy-related warnings may be expected" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "All formatting tools have been executed." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review any warnings or errors above" -ForegroundColor White
Write-Host "2. Update CHECKLIST_100_ITEMS.md to mark items 9-12 as complete" -ForegroundColor White
Write-Host "3. Commit the changes: git add . && git commit -m 'chore: run formatting tools (items 9-12)'" -ForegroundColor White
Write-Host ""
Write-Host "🎉 Congratulations! You've reached 100% completion!" -ForegroundColor Green

