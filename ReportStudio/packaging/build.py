"""Build the Report Studio desktop app into a standalone executable.

Usage (from the repo root, in the project venv):
    python packaging/build.py            # build with whatever's in packaging/bin/
    python packaging/build.py --fetch-tectonic   # also download Tectonic first

This is a thin, readable wrapper around PyInstaller so the build is one command and
the Tectonic bundling story is explicit. It does NOT try to be clever: it shells out
to `pyinstaller` with the spec and reports where the output landed.

Why a script and not just `pyinstaller`:
  - it can optionally fetch the right Tectonic binary into packaging/bin/ so the
    resulting exe compiles PDFs with zero user setup (the whole point of the pivot);
  - it fails loudly with actionable messages when a prerequisite is missing.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "ReportStudio.spec"
BIN_DIR = ROOT / "packaging" / "bin"

# Tectonic release assets (self-contained single binary). Pinned to a known-good tag.
_TECTONIC_VERSION = "0.15.0"
_TECTONIC_ASSETS = {
    ("Windows", "AMD64"): f"tectonic-{_TECTONIC_VERSION}-x86_64-pc-windows-msvc.zip",
    ("Darwin", "arm64"): f"tectonic-{_TECTONIC_VERSION}-aarch64-apple-darwin.tar.gz",
    ("Darwin", "x86_64"): f"tectonic-{_TECTONIC_VERSION}-x86_64-apple-darwin.tar.gz",
    ("Linux", "x86_64"): f"tectonic-{_TECTONIC_VERSION}-x86_64-unknown-linux-gnu.tar.gz",
}
_TECTONIC_BASE = (
    "https://github.com/tectonic-typesetting/tectonic/releases/download/"
    f"tectonic%40{_TECTONIC_VERSION}/"
)


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(
            f"'{tool}' not found on PATH. Install the dev extras first:\n"
            f"    pip install -e \".[app,dev]\""
        )


def fetch_tectonic() -> None:
    """Download the platform-appropriate Tectonic binary into packaging/bin/.

    Best-effort: on an unknown platform or a network failure we warn and continue —
    the app still builds and falls back to a system TeX engine at runtime.
    """
    import io
    import tarfile
    import urllib.request
    import zipfile

    key = (platform.system(), platform.machine())
    asset = _TECTONIC_ASSETS.get(key)
    if not asset:
        print(f"! No known Tectonic asset for {key}; skipping. "
              f"The exe will rely on a system TeX engine.")
        return

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    url = _TECTONIC_BASE + asset
    print(f"Downloading Tectonic {_TECTONIC_VERSION} for {key}…\n  {url}")
    try:
        raw = urllib.request.urlopen(url, timeout=120).read()
    except Exception as e:
        print(f"! Tectonic download failed ({e}); skipping bundling.")
        return

    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for name in z.namelist():
                if name.endswith("tectonic.exe") or name == "tectonic":
                    (BIN_DIR / Path(name).name).write_bytes(z.read(name))
    else:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
            for m in t.getmembers():
                if Path(m.name).name == "tectonic":
                    data = t.extractfile(m).read()
                    out = BIN_DIR / "tectonic"
                    out.write_bytes(data)
                    out.chmod(0o755)
    print(f"  -> bundled into {BIN_DIR}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Report Studio .exe")
    ap.add_argument("--fetch-tectonic", action="store_true",
                    help="download Tectonic into packaging/bin/ before building")
    ap.add_argument("--clean", action="store_true", help="clean build/ and dist/ first")
    args = ap.parse_args()

    _require("pyinstaller")

    if args.fetch_tectonic:
        fetch_tectonic()
    elif not BIN_DIR.exists() or not any(BIN_DIR.iterdir()):
        print("i No Tectonic binary in packaging/bin/. Building anyway; the app will "
              "use a system TeX engine if present. Run with --fetch-tectonic to bundle "
              "a self-contained PDF engine.")

    if args.clean:
        for d in ("build", "dist"):
            shutil.rmtree(ROOT / d, ignore_errors=True)

    print("Running PyInstaller…")
    proc = subprocess.run(
        ["pyinstaller", "--noconfirm", str(SPEC)], cwd=str(ROOT)
    )
    if proc.returncode != 0:
        return proc.returncode

    out = ROOT / "dist" / "ReportStudio"
    exe = out / ("ReportStudio.exe" if platform.system() == "Windows" else "ReportStudio")
    print("\nBuild complete.")
    print(f"  App folder: {out}")
    print(f"  Executable: {exe}")
    print("Ship the whole 'ReportStudio' folder (onedir build).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
