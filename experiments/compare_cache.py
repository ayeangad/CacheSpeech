import time
import torch

from cachespeech.audio import load_audio, extract_features
from cachespeech.model import CacheSpeechModel
from cachespeech.evaluation import run_cache_aware, run_no_cache, WER_TRANSFORM
from jiwer import wer


AUDIO_PATH = "audio/clean-1.wav"

REFERENCE_TEXT = "Hello how are you how was your morning"

LOOKAHEAD = [70, 6]


def benchmark_method(
    method,
    model,
    features,
    audio_duration,
):
    """Run one method and measure wall-clock inference time."""

    if model.device.startswith("cuda"):
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    result = method(
        model,
        features,
    )

    if model.device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    result["inference_time"] = elapsed

    result["rtf"] = elapsed / audio_duration

    result["wer"] = wer(
        WER_TRANSFORM(REFERENCE_TEXT),
        WER_TRANSFORM(result["final_text"]),
    )

    return result


def print_result(
    name,
    result,
):
    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(
        "Final:",
        repr(result["final_text"]),
    )

    print(
        "WER:",
        round(result["wer"], 4),
    )

    print(
        "Chunks:",
        result["chunks"],
    )

    print(
        "Encoded frames:",
        result["encoded_frames"],
    )

    print(
        "Input frames processed:",
        result["input_frames_processed"],
    )

    print(
        "Inference:",
        round(result["inference_time"], 4),
        "s",
    )

    print(
        "RTF:",
        round(result["rtf"], 4),
    )


def main():

    print("Loading model...")

    model = CacheSpeechModel()

    model.set_lookahead(LOOKAHEAD)

    print()
    print("Loading audio...")

    waveform, sample_rate = load_audio(
        AUDIO_PATH,
    )

    features, _ = extract_features(
        model,
        waveform,
        sample_rate,
    )

    audio_duration = (
        waveform.shape[-1] / sample_rate
    )

    print(
        "Features:",
        features.shape,
    )

    print(
        "Audio duration:",
        round(audio_duration, 3),
        "s",
    )

    print()
    print("=" * 70)
    print("CACHE VS NO-CACHE")
    print("=" * 70)

    print()
    print("Lookahead:", LOOKAHEAD)

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    print()
    print("Warming up...")

    benchmark_method(
        run_cache_aware,
        model,
        features,
        audio_duration,
    )

    benchmark_method(
        run_no_cache,
        model,
        features,
        audio_duration,
    )

    print("Warm-up complete.")

    # ---------------------------------------------------------
    # CacheSpeech
    # ---------------------------------------------------------

    cached = benchmark_method(
        run_cache_aware,
        model,
        features,
        audio_duration,
    )

    print_result(
        "CACHE-AWARE",
        cached,
    )

    # ---------------------------------------------------------
    # No-cache
    # ---------------------------------------------------------

    no_cache = benchmark_method(
        run_no_cache,
        model,
        features,
        audio_duration,
    )

    print_result(
        "NO-CACHE",
        no_cache,
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print()

    print(
        f"{'Metric':<32}"
        f"{'CacheSpeech':<18}"
        f"{'No-cache':<18}"
    )

    print("-" * 70)

    print(
        f"{'WER':<32}"
        f"{cached['wer']:<18.4f}"
        f"{no_cache['wer']:<18.4f}"
    )

    print(
        f"{'Inference time (s)':<32}"
        f"{cached['inference_time']:<18.4f}"
        f"{no_cache['inference_time']:<18.4f}"
    )

    print(
        f"{'RTF':<32}"
        f"{cached['rtf']:<18.4f}"
        f"{no_cache['rtf']:<18.4f}"
    )

    print(
        f"{'Encoded frames':<32}"
        f"{cached['encoded_frames']:<18}"
        f"{no_cache['encoded_frames']:<18}"
    )

    print(
        f"{'Input frames processed':<32}"
        f"{cached['input_frames_processed']:<18}"
        f"{no_cache['input_frames_processed']:<18}"
    )

    # ---------------------------------------------------------
    # Derived metrics
    # ---------------------------------------------------------

    if cached["inference_time"] > 0:

        speedup = (
            no_cache["inference_time"]
            / cached["inference_time"]
        )

    else:
        speedup = float("inf")

    if no_cache["input_frames_processed"] > 0:

        input_reduction = (
            1
            - (
                cached["input_frames_processed"]
                / no_cache["input_frames_processed"]
            )
        ) * 100

    else:
        input_reduction = 0.0

    print()

    print(
        f"CacheSpeech speedup: "
        f"{speedup:.2f}x"
    )

    print(
        f"Input-frame reduction: "
        f"{input_reduction:.2f}%"
    )


if __name__ == "__main__":
    main()