#!/usr/bin/env python3
"""
Voice Digital Signature — Verification

Checks whether an audio file matches a stored voice signature.

Usage:
    python verify.py --signature john_doe.npy audio.mp3
    python verify.py --signature john_doe.npy test.wav --threshold 0.75
    python verify.py --signature john_doe.npy *.wav   # batch verify
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from sign import embed_file, ensure_wav16k, CHUNK_SEC

DEFAULT_THRESHOLD = 0.70


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def verify_file(wav_path: Path, signature: np.ndarray, chunk_sec: float, device: str) -> dict:
    """
    Extract embedding from audio and compare against signature.
    Returns dict with similarity, match, and per-chunk details.
    """
    chunks_emb: list[np.ndarray] = []
    from sign import chunk_audio, embed_chunk, MIN_CHUNK_SEC
    import soundfile as sf

    audio_chunks = chunk_audio(wav_path, chunk_sec)
    if not audio_chunks:
        raise ValueError(f"No usable audio in {wav_path}")

    sims: list[float] = []
    for chunk in audio_chunks:
        emb = embed_chunk(chunk, device)
        sim = cosine_similarity(emb, signature)
        sims.append(sim)

    # Overall similarity: mean of per-chunk similarities
    mean_sim = float(np.mean(sims))
    max_sim = float(np.max(sims))
    min_sim = float(np.min(sims))

    return {
        "mean_similarity": round(mean_sim, 4),
        "max_similarity": round(max_sim, 4),
        "min_similarity": round(min_sim, 4),
        "chunks": len(sims),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify whether an audio file matches a voice signature.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python verify.py --signature john_doe.npy audio.mp3\n"
            "  python verify.py --signature john_doe.npy test.wav --threshold 0.75\n"
        ),
    )
    parser.add_argument("audio", nargs="+", help="Audio file(s) to verify")
    parser.add_argument("--signature", "-s", required=True, help="Path to .npy signature file")
    parser.add_argument(
        "--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for MATCH (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--chunk", "-c", type=float, default=CHUNK_SEC,
        help=f"Chunk length in seconds (default: {CHUNK_SEC})",
    )
    parser.add_argument("--device", default=None, help="Device: 'cuda' or 'cpu' (default: auto-detect)")
    args = parser.parse_args()

    sig_path = Path(args.signature)
    if not sig_path.exists():
        print(f"Error: signature file not found: {sig_path}", file=sys.stderr)
        sys.exit(1)

    signature = np.load(str(sig_path)).astype(np.float32)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device    : {device}")
    print(f"Signature : {sig_path.name}  ({len(signature)}D)")
    print(f"Threshold : {args.threshold}")

    audio_paths = [Path(p) for p in args.audio]
    missing = [p for p in audio_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    results = []
    with tempfile.TemporaryDirectory(prefix="voice_verify_") as tmpdir:
        tmpdir = Path(tmpdir)
        for path in audio_paths:
            print(f"\n[{path.name}]")
            try:
                wav = ensure_wav16k(path, tmpdir)
                result = verify_file(wav, signature, args.chunk, device)
                match = result["mean_similarity"] >= args.threshold
                verdict = "MATCH" if match else "NO MATCH"
                result["file"] = path.name
                result["match"] = match
                results.append(result)
                print(f"  Similarity (mean): {result['mean_similarity']:.4f}")
                print(f"  Similarity (max) : {result['max_similarity']:.4f}")
                print(f"  Similarity (min) : {result['min_similarity']:.4f}")
                print(f"  Chunks           : {result['chunks']}")
                print(f"  Verdict          : {verdict}")
            except Exception as exc:
                print(f"  Error: {exc}", file=sys.stderr)

    if len(results) > 1:
        print(f"\n{'='*55}")
        print(f"{'File':<30} {'Similarity':>10} {'Result':>12}")
        print("-" * 55)
        for r in results:
            verdict = "MATCH" if r["match"] else "NO MATCH"
            print(f"{r['file']:<30} {r['mean_similarity']:>10.4f} {verdict:>12}")

    # Exit code: 0 if all match, 1 if any don't match
    if not all(r["match"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
