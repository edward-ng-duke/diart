"""Build-time one-shot downloader for all models declared in models.json.

Used by the Docker build to bundle every supported model into the image so the
runtime container never needs network or HF credentials.

Usage:
    HF_TOKEN=hf_xxx python docker/download_models.py /opt/diart/models
"""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from pyannote.audio import Model


def _is_non_empty(target: Path) -> bool:
    """Return True if `target` already has content (dir w/ entries OR existing file)."""
    if not target.exists():
        return False
    if target.is_dir():
        return any(target.iterdir())
    return target.stat().st_size > 0


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(__file__).parent / "models.json"
    if not manifest_path.exists():
        print(f"[error] manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "[error] HF_TOKEN environment variable is required for build-time downloads.",
            file=sys.stderr,
        )
        sys.exit(1)

    failures: list[str] = []

    for spec in manifest:
        hf_name = spec["hf_name"]
        backend = spec["backend"]
        target = out_dir / spec["local_path"]

        if _is_non_empty(target):
            print(f"[skip] {hf_name} (already present at {target})")
            continue

        print(f"[download] {hf_name} -> {target}")
        try:
            if backend in ("pyannote", "speechbrain", "nemo"):
                snapshot_download(
                    repo_id=hf_name,
                    local_dir=str(target),
                    token=token,
                )
            elif backend == "onnx":
                target.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=hf_name,
                    filename="model.onnx",
                    local_dir=str(target.parent),
                    token=token,
                )
            else:
                print(f"[error] unknown backend: {backend!r} for {hf_name}", file=sys.stderr)
                failures.append(hf_name)
                continue
        except Exception as exc:  # noqa: BLE001 - surface any download error
            print(f"[error] failed to download {hf_name}: {exc}", file=sys.stderr)
            failures.append(hf_name)
            continue

        # Smoke-load verification for pyannote checkpoints — catches truncated
        # downloads or missing config files at build time rather than runtime.
        if backend == "pyannote":
            try:
                Model.from_pretrained(str(target), use_auth_token=False)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[error] smoke-load failed for {hf_name} at {target}: {exc}",
                    file=sys.stderr,
                )
                failures.append(hf_name)

    if failures:
        print(
            f"[fatal] {len(failures)} model(s) failed: {', '.join(failures)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[done] all {len(manifest)} models present under {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python download_models.py <output_dir>", file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]))
