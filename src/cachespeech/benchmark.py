from __future__ import annotations

import time

import soundfile as sf
import torch
import torchaudio

from .model import CacheSpeechModel
from .streaming import CacheSpeechStream

REFERENCE_TEXT = (
    "Hi How are you how was your morning"
)

def load_features(
    model: CacheSpeechModel,
    path: str,
) -> tuple[torch.Tensor, int, float]:
    """Load audio and convert it to NeMo-compatible Mel features."""

    audio, sample_rate = sf.read(path, dtype="float32")

    waveform = torch.from_numpy(audio).unsqueeze(0)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=16000,
        )

    duration = waveform.shape[-1] / 16000

    features, feature_lengths = model.model.preprocessor(
        input_signal=waveform.to(model.device),
        length=torch.tensor(
            [waveform.shape[-1]],
            device=model.device,
        ),
    )

    return features, feature_lengths.item(), duration


def benchmark(
    audio_path: str = "audio/audio-example-1.wav",
    chunk_size: int = 105,
) -> dict:
    """Benchmark cache-aware streaming inference."""

    model = CacheSpeechModel()
    stream = CacheSpeechStream(model)

    features, valid_frames, duration = load_features(
        model,
        audio_path,
    )
    
    cfg = model.model.encoder.cfg
    if isinstance(cfg.pre_encode_cache_size, list):
        pre_encode = cfg.pre_encode_cache_size[1]
    else:
        pre_encode = cfg.pre_encode_cache_size

    input_chunk_size = chunk_size + pre_encode
    features = F.pad(features, (pre_encode, 0))

    if model.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()

    start_time = time.perf_counter()

    num_chunks = 0
    final_text = ""
    start = 0

    while start < features.shape[-1] - pre_encode:
        end = min(start + input_chunk_size, features.shape[-1])

        chunk = features[:, :, start:end]

        actual_length = chunk.shape[-1]
        if actual_length == 0:
            break
            
        if actual_length < input_chunk_size:
            pad_amount = input_chunk_size - actual_length
            chunk = F.pad(chunk, (0, pad_amount))

        final_text = stream.step(
            chunk,
            feature_length=chunk.shape[-1],
        )

        num_chunks += 1
        start += chunk_size

    if model.device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    rtf = elapsed / duration

    peak_memory_mb = None

    if model.device == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
        )

    return {
    "audio_duration_s": duration,
    "valid_feature_frames": valid_frames,
    "chunk_size": chunk_size,
    "num_chunks": num_chunks,
    "inference_time_s": elapsed,
    "rtf": rtf,
    "peak_gpu_memory_mb": peak_memory_mb,
    "final_text": final_text,
    "reference_text": REFERENCE_TEXT,
    }


if __name__ == "__main__":
    result = benchmark()

    print()
    print("=" * 60)
    print("CacheSpeech Baseline")
    print("=" * 60)

    print(f"Audio duration:       {result['audio_duration_s']:.3f}s")
    print(f"Feature frames:       {result['valid_feature_frames']}")
    print(f"Chunk size:           {result['chunk_size']}")
    print(f"Chunks:               {result['num_chunks']}")
    print(f"Inference time:       {result['inference_time_s']:.3f}s")
    print(f"RTF:                  {result['rtf']:.4f}")

    if result["peak_gpu_memory_mb"] is not None:
        print(
            f"Peak GPU memory:      "
            f"{result['peak_gpu_memory_mb']:.1f} MB"
        )

    print(f"Final hypothesis:     {result['final_text']!r}")
    print("=" * 60)