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


def _is_complete(target: Path, backend: str) -> bool:
    """Return True if a previous download for ``target`` finished successfully.

    A bare existence check is too lax: a partial/aborted ``snapshot_download``
    can leave a directory containing only ``LICENSE``/``README.md`` while the
    actual weights are missing. We require a backend-specific marker.
    """
    if not target.exists():
        return False
    if backend == "onnx":
        return target.is_file() and target.stat().st_size > 0
    if backend == "pyannote":
        return (target / "pytorch_model.bin").exists()
    if backend == "speechbrain":
        # speechbrain HF repos always include hyperparams.yaml + an embedding/classifier ckpt
        return (target / "hyperparams.yaml").exists()
    if backend == "nemo":
        # nemo bundles a single .nemo archive
        return any(target.glob("*.nemo")) if target.is_dir() else False
    # Unknown backend: fall back to "any content".
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

        if _is_complete(target, backend):
            print(f"[skip] {hf_name} (already present at {target})")
            continue
        # Clean up any partial residue from a prior aborted download so
        # ``snapshot_download`` doesn't re-use stale state.
        if target.exists() and not _is_complete(target, backend):
            print(f"[clean] removing partial download at {target}")
            if target.is_dir():
                import shutil
                shutil.rmtree(target)
            else:
                target.unlink()

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
                # ONNX repos may use any filename (e.g. "speaker-embedding.onnx").
                # Honour the manifest's "filename" override when present.
                filename = spec.get("filename", "model.onnx")
                hf_hub_download(
                    repo_id=hf_name,
                    filename=filename,
                    local_dir=str(target.parent),
                    token=token,
                )
                # If the resolved filename differs from the manifest's basename,
                # symlink/rename so the runtime registry can find it at the
                # exact path the manifest declares.
                resolved = target.parent / filename
                if resolved.exists() and resolved != target:
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    target.symlink_to(resolved.name)
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
        # ``Model.from_pretrained`` expects either an HF repo id or a path to
        # the checkpoint file itself (not the snapshot directory), so we look
        # for ``pytorch_model.bin`` inside the snapshot.
        if backend == "pyannote":
            checkpoint = target / "pytorch_model.bin"
            if not checkpoint.exists():
                # Fall back to any *.bin in the snapshot.
                bins = list(target.glob("*.bin"))
                checkpoint = bins[0] if bins else None
            if checkpoint is None:
                print(
                    f"[error] no checkpoint *.bin found in {target} for {hf_name}",
                    file=sys.stderr,
                )
                failures.append(hf_name)
            else:
                try:
                    Model.from_pretrained(str(checkpoint), use_auth_token=False)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[error] smoke-load failed for {hf_name} at {checkpoint}: {exc}",
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
