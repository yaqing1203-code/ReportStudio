#!/usr/bin/env bash
# Build script for ReportStudio.exe with automatic cleanup
# Usage: ./build.sh

set -e  # Exit on error

echo "========================================"
echo "Building ReportStudio.exe"
echo "========================================"
echo ""

# Activate Python 3.11 conda environment and build
echo "[1/3] Running PyInstaller..."
conda run -n py311 python -m PyInstaller packaging/ReportStudio.spec --clean --noconfirm

echo ""
echo "[2/3] Verifying build output..."

# Verify the executable was created
if [ ! -f "dist/ReportStudio/ReportStudio.exe" ]; then
    echo "ERROR: Build failed - ReportStudio.exe not found in dist/"
    exit 1
fi

# Show build summary
EXE_SIZE=$(du -h dist/ReportStudio/ReportStudio.exe | cut -f1)
TOTAL_SIZE=$(du -sh dist/ReportStudio | cut -f1)

echo "✓ Build successful!"
echo "  Executable: dist/ReportStudio/ReportStudio.exe ($EXE_SIZE)"
echo "  Total package: $TOTAL_SIZE"

# Check if Tectonic is bundled
if [ -f "dist/ReportStudio/_internal/bin/tectonic.exe" ]; then
    TECTONIC_SIZE=$(du -h dist/ReportStudio/_internal/bin/tectonic.exe | cut -f1)
    echo "  Tectonic bundled: $TECTONIC_SIZE"
fi

echo ""
echo "[3/3] Cleaning up build artifacts..."

# Remove the build/ directory (intermediate files only, not needed for distribution)
if [ -d "build" ]; then
    rm -rf build
    echo "✓ Removed build/ directory (intermediate files)"
fi

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo ""
echo "Distribution package: dist/ReportStudio/"
echo ""
echo "To deploy:"
echo "  1. Copy the entire dist/ReportStudio/ folder to target machines"
echo "  2. Run ReportStudio.exe"
echo ""
echo "Note: The build/ folder has been removed automatically."
echo "      Only dist/ReportStudio/ is needed for distribution."
