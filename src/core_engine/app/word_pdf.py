"""Use an installed office renderer without changing the generated DOCX package."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

_WORD_LOCK = threading.Lock()
_WORD_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$word = $null
$doc = $null
$owned = $false
function Invoke-Office([scriptblock]$Action) {
  for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try { return & $Action }
    catch {
      if ($_.Exception.Message -notmatch 'rejected|busy|0x80010001|0x8001010A') { throw }
      Start-Sleep -Milliseconds 250
    }
  }
  throw 'Word remained busy during PDF conversion.'
}
try {
  $oldIds = @(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
  $word = New-Object -ComObject Word.Application
  $newProcesses = @(Get-Process WINWORD -ErrorAction SilentlyContinue | Where-Object { $oldIds -notcontains $_.Id })
  if ($newProcesses.Count -ne 1) { throw 'A separate Word instance could not be identified.' }
  $wordProcessId = $newProcesses[0].Id
  $owned = $true
  [IO.File]::WriteAllText($env:RS_TEMPLATE_PID, [string]$wordProcessId)
  Invoke-Office { $word.Visible = $false; $word.DisplayAlerts = 0; $word.AutomationSecurity = 3 }
  Invoke-Office { $script:doc = $word.Documents.Open($env:RS_TEMPLATE_DOCX, $false, $true, $false) }
  if ($null -eq $doc) { throw 'Word did not return an opened document.' }
  Invoke-Office { $doc.Fields.Update() } | Out-Null
  Invoke-Office { $doc.ExportAsFixedFormat($env:RS_TEMPLATE_PDF, 17) }
} finally {
  if ($doc -ne $null) { try { Invoke-Office { $doc.Close(0) } } catch {} }
  if ($owned -and $word -ne $null) {
    try { Invoke-Office { $word.Quit(0) } }
    catch { Stop-Process -Id $wordProcessId -Force -ErrorAction SilentlyContinue }
  }
}
"""


def convert_docx_pdf(source: Path, destination: Path, *, timeout: float = 120) -> None:
    source, destination = source.resolve(), destination.resolve()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    destination.unlink(missing_ok=True)
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice and os.name == "nt":
        candidate = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "LibreOffice/program/soffice.exe"
        soffice = str(candidate) if candidate.is_file() else None
    if soffice:
        with tempfile.TemporaryDirectory(prefix="rs-template-pdf-") as temp:
            root = Path(temp)
            subprocess.run([soffice, "--headless", "-env:UserInstallation=" + (root / "profile").as_uri(),
                "--convert-to", "pdf:writer_pdf_Export", "--outdir", str(root), str(source)],
                check=True, capture_output=True, timeout=timeout, creationflags=flags)
            shutil.copyfile(root / (source.stem + ".pdf"), destination)
    elif os.name == "nt":
        powershell = shutil.which("powershell.exe")
        if not powershell:
            raise RuntimeError("Word or LibreOffice is required for template PDF export")
        with _WORD_LOCK, tempfile.TemporaryDirectory(prefix="rs-template-word-") as temp:
            pid_file = Path(temp) / "word.pid"
            env = {**os.environ, "RS_TEMPLATE_DOCX": str(source), "RS_TEMPLATE_PDF": str(destination),
                   "RS_TEMPLATE_PID": str(pid_file)}
            try:
                subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", _WORD_SCRIPT],
                    check=True, capture_output=True, timeout=timeout, env=env, creationflags=flags)
            except subprocess.TimeoutExpired:
                # Only the new instance whose PID this worker recorded is ours.
                if pid_file.exists() and pid_file.read_text().strip().isdigit():
                    subprocess.run(["taskkill.exe", "/PID", pid_file.read_text().strip(), "/F"],
                        check=False, capture_output=True, timeout=10, creationflags=flags)
                raise
    else:
        raise RuntimeError("LibreOffice is required for template PDF export")
    if not destination.is_file() or destination.read_bytes()[:5] != b"%PDF-":
        destination.unlink(missing_ok=True)
        raise RuntimeError("Office renderer did not produce a valid PDF")
