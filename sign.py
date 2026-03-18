#!/usr/bin/env python3
"""
Voice Digital Signature — Enrollment

Creates a speaker voice signature from one or more audio files (mp3/wav).
Compatible with the nemo-diarization identify_speakers() format (.npy files).

Strategy:
  1. Convert each file to 16kHz mono WAV (ffmpeg)
  2. Chunk audio into ~CHUNK_SEC-second segments
  3. Extract TitaNet embedding per chunk
  4. Average all embeddings → L2-normalize → speaker signature

Usage:
    python sign.py --name "John Doe" audio1.mp3 audio2.wav ...
    python sign.py --name "John Doe" --output signatures/ recordings/*.wav
    python sign.py --name "John Doe" --chunk 5 audio.mp3  # 5-sec chunks
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Default chunk length in seconds for embedding extraction
CHUNK_SEC = 3.0
# Minimum chunk length to bother extracting (avoid noisy tiny trailing chunks)
MIN_CHUNK_SEC = 1.0


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def convert_to_wav16k(src: Path, dst: Path) -> None:
    """Convert any audio file to 16kHz mono WAV using ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src}:\n{result.stderr}")


def needs_conversion(path: Path) -> bool:
    """Return True if the file is not already 16kHz mono WAV."""
    if path.suffix.lower() != ".wav":
        return True
    info = sf.info(str(path))
    return info.samplerate != 16000 or info.channels != 1


def ensure_wav16k(path: Path, tmpdir: Path) -> Path:
    """Return a 16kHz mono WAV path, converting if necessary."""
    if not needs_conversion(path):
        return path
    dst = tmpdir / f"{path.stem}_16k.wav"
    print(f"  Converting {path.name} → 16kHz mono WAV...")
    convert_to_wav16k(path, dst)
    return dst


def chunk_audio(wav_path: Path, chunk_sec: float) -> list[np.ndarray]:
    """Split a WAV into fixed-length chunks. Returns list of float32 arrays."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    chunk_len = int(chunk_sec * sr)
    min_len = int(MIN_CHUNK_SEC * sr)
    chunks = []
    for start in range(0, len(audio), chunk_len):
        chunk = audio[start : start + chunk_len]
        if len(chunk) >= min_len:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# TitaNet embedding (NeMo)
# ---------------------------------------------------------------------------

_model = None


def _load_model(device: str):
    global _model
    if _model is None:
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except ImportError:
            print("Error: NeMo not installed.  Run: pip install nemo_toolkit[asr]", file=sys.stderr)
            sys.exit(1)
        print("Loading TitaNet-Large (first run downloads ~90 MB)...")
        _model = EncDecSpeakerLabelModel.from_pretrained("titanet_large")
        _model.eval()
    return _model.to(device)


def embed_chunk(audio: np.ndarray, device: str) -> np.ndarray:
    """Return a L2-normalized speaker embedding vector for a single audio chunk."""
    model = _load_model(device)
    t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    length = torch.tensor([len(audio)]).to(device)
    with torch.no_grad():
        _, emb = model(input_signal=t, input_signal_length=length)
    v = emb.cpu().numpy().squeeze()
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v


def embed_file(wav_path: Path, chunk_sec: float, device: str) -> tuple[np.ndarray, int]:
    """
    Extract embeddings from all chunks of a WAV file.
    Returns (mean_embedding, num_chunks).
    """
    chunks = chunk_audio(wav_path, chunk_sec)
    if not chunks:
        raise ValueError(f"No usable audio in {wav_path}")
    embeddings = [embed_chunk(c, device) for c in chunks]
    mean_emb = np.mean(embeddings, axis=0)
    return mean_emb, len(chunks)


# ---------------------------------------------------------------------------
# Signature creation
# ---------------------------------------------------------------------------

def build_signature(audio_paths: list[Path], chunk_sec: float, device: str) -> tuple[np.ndarray, dict]:
    """
    Process all audio files and produce a single aggregated voice signature.

    Returns:
        signature: L2-normalized float32 ndarray (TitaNet embedding space)
        stats:     metadata dict about the enrollment
    """
    all_embeddings: list[np.ndarray] = []
    file_stats: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="voice_sig_") as tmpdir:
        tmpdir = Path(tmpdir)
        for path in audio_paths:
            print(f"\n[{path.name}]")
            try:
                wav = ensure_wav16k(path, tmpdir)
                emb, n_chunks = embed_file(wav, chunk_sec, device)
                all_embeddings.append(emb)
                duration = sf.info(str(wav)).duration
                file_stats.append({"file": path.name, "chunks": n_chunks, "duration_sec": round(duration, 1)})
                print(f"  {n_chunks} chunks extracted  ({duration:.1f}s)")
            except Exception as exc:
                print(f"  Warning: skipping {path.name} — {exc}", file=sys.stderr)

    if not all_embeddings:
        raise RuntimeError("No embeddings could be extracted from any of the provided files.")

    # Aggregate: weighted mean (all files contribute equally regardless of chunk count)
    signature = np.mean(all_embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(signature)
    signature = signature / norm if norm > 1e-8 else signature

    total_chunks = sum(f["chunks"] for f in file_stats)
    total_dur = sum(f["duration_sec"] for f in file_stats)
    stats = {
        "files": len(file_stats),
        "total_chunks": total_chunks,
        "total_duration_sec": round(total_dur, 1),
        "chunk_sec": chunk_sec,
        "embedding_dim": len(signature),
        "files_detail": file_stats,
    }
    return signature, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a voice digital signature from audio files using NeMo TitaNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python sign.py --name 'John Doe' recording1.mp3 recording2.wav\n"
            "  python sign.py --name 'John Doe' --output ./signatures/ *.wav\n"
            "  python sign.py --name 'John Doe' --chunk 5 long_speech.mp3\n"
        ),
    )
    parser.add_argument("audio", nargs="+", help="Audio file(s) (mp3, wav, etc.)")
    parser.add_argument("--name", "-n", required=True, help="Speaker name (used in output filename)")
    parser.add_argument(
        "--output", "-o", default=".",
        help="Output directory for the .npy signature file (default: current directory)",
    )
    parser.add_argument(
        "--chunk", "-c", type=float, default=CHUNK_SEC,
        help=f"Chunk length in seconds for embedding extraction (default: {CHUNK_SEC})",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: 'cuda' or 'cpu' (default: auto-detect)",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                device = "cuda"
            except RuntimeError:
                print("Warning: CUDA disponibil dar incompatibil cu GPU-ul curent — folosim CPU.", file=sys.stderr)
                device = "cpu"
        else:
            device = "cpu"
    print(f"Device: {device}")

    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}
    raw_paths = [Path(p) for p in args.audio]
    audio_paths: list[Path] = []
    for p in raw_paths:
        if p.is_dir():
            found = sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)
            if not found:
                print(f"Warning: no audio files found in {p}", file=sys.stderr)
            else:
                print(f"  Folder {p}: {len(found)} fișiere găsite")
            audio_paths.extend(found)
        else:
            audio_paths.append(p)

    missing = [p for p in audio_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: file not found: {p}", file=sys.stderr)
        sys.exit(1)
    if not audio_paths:
        print("Error: niciun fișier audio de procesat.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEnrolling speaker: {args.name}")
    print(f"Audio files: {len(audio_paths)}")
    print(f"Chunk length: {args.chunk}s")

    signature, stats = build_signature(audio_paths, args.chunk, device)

    # Save signature as .npy (compatible with nemo-diarization identify_speakers)
    safe_name = args.name.replace(" ", "_").lower()
    sig_path = output_dir / f"{safe_name}.npy"
    np.save(str(sig_path), signature)

    # Save metadata alongside
    meta_path = output_dir / f"{safe_name}_meta.json"
    meta = {"speaker": args.name, **stats}
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n{'='*50}")
    print(f"  Speaker  : {args.name}")
    print(f"  Files    : {stats['files']}")
    print(f"  Chunks   : {stats['total_chunks']}")
    print(f"  Duration : {stats['total_duration_sec']}s total")
    print(f"  Embedding: {stats['embedding_dim']}D (TitaNet-Large)")
    print(f"\n  Signature: {sig_path}")
    print(f"  Metadata : {meta_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
