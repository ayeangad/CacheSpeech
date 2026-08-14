import csv
import json
import time
from pathlib import Path

import torch
from jiwer import wer

from cachespeech.audio import extract_features, load_audio
from cachespeech.model import CacheSpeechModel
from cachespeech.evaluation import (
    run_cache_aware,
    run_no_cache,
    warm_up,
    mean,
    std,
    WER_TRANSFORM,
)


AUDIO_DIR = Path("audio")
RESULTS_DIR = Path("experiments")

RUNS = 5
WARMUP_RUNS = 1

LOOKAHEADS = [
    [70, 13],
    [70, 6],
    [70, 1],
    [70, 0],
]


DATASET = {
    "clean-1.wav": (
        "Hello how are you how was your morning"
    ),
    "clean-2.wav": (
        "Hello I'm doing good my evening was amazing"
    ),
    "noise 2.wav": (
        "Hello this is noisy example audio second"
    ),
    "women-1.wav": (
        "Hi I miss you you're really cute and you're really "
        "hot and I miss you so much and you're so sweet I miss "
        "you okay that's longer than oh oh shit okay I'll stop"
    ),
}


def synchronize(model: CacheSpeechModel) -> None:
    if model.device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark_method(
    method,
    model: CacheSpeechModel,
    features: torch.Tensor,
    reference: str,
    audio_duration: float,
) -> dict:
    """
    Run one method multiple times and measure wall-clock
    inference time.
    """

    # GPU warm-up.
    for _ in range(WARMUP_RUNS):
        method(model, features)

    synchronize(model)

    times = []
    rtfs = []
    wers = []

    last_result = None

    for run in range(RUNS):

        synchronize(model)

        start_time = time.perf_counter()

        result = method(
            model,
            features,
        )

        synchronize(model)

        elapsed = time.perf_counter() - start_time

        normalized_reference = WER_TRANSFORM(reference)

        normalized_hypothesis = WER_TRANSFORM(
            result["final_text"]
        )

        error_rate = wer(
            normalized_reference,
            normalized_hypothesis,
        )

        rtf = elapsed / audio_duration

        times.append(elapsed)
        rtfs.append(rtf)
        wers.append(error_rate)

        last_result = result

    return {
        "wer_mean": mean(wers),
        "wer_std": std(wers),
        "time_mean": mean(times),
        "time_std": std(times),
        "rtf_mean": mean(rtfs),
        "rtf_std": std(rtfs),
        "chunks": last_result["chunks"],
        "encoded_frames": last_result["encoded_frames"],
        "input_frames": last_result["input_frames"],
        "final_text": last_result["final_text"],
    }


def main():

    print("Loading model...")
    model = CacheSpeechModel()

    results = []

    print()
    print("=" * 100)
    print("DATASET CACHE VS NO-CACHE BENCHMARK")
    print("=" * 100)
    print()
    print(f"Runs: {RUNS}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Audio files: {len(DATASET)}")
    print(f"Lookaheads: {len(LOOKAHEADS)}")
    print()

    for audio_name, reference in DATASET.items():

        audio_path = AUDIO_DIR / audio_name

        print()
        print("#" * 100)
        print(f"AUDIO: {audio_name}")
        print("#" * 100)

        print("Loading audio...")

        waveform, sample_rate = load_audio(
            str(audio_path),
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
            f"Duration: {audio_duration:.3f}s | "
            f"Features: {tuple(features.shape)}"
        )

        for lookahead in LOOKAHEADS:

            print()
            print("-" * 100)
            print(f"LOOKAHEAD: {lookahead}")
            print("-" * 100)

            model.set_lookahead(lookahead)

            print("  Benchmarking CACHE-AWARE...")

            cache_result = benchmark_method(
                run_cache_aware,
                model,
                features,
                reference,
                audio_duration,
            )

            print(
                f"    WER={cache_result['wer_mean']:.4f} "
                f"RTF={cache_result['rtf_mean']:.4f} "
                f"time={cache_result['time_mean']:.4f}s "
                f"encoded={cache_result['encoded_frames']}"
            )

            print("  Benchmarking NO-CACHE...")

            no_cache_result = benchmark_method(
                run_no_cache,
                model,
                features,
                reference,
                audio_duration,
            )

            print(
                f"    WER={no_cache_result['wer_mean']:.4f} "
                f"RTF={no_cache_result['rtf_mean']:.4f} "
                f"time={no_cache_result['time_mean']:.4f}s "
                f"encoded={no_cache_result['encoded_frames']}"
            )

            input_reduction = (
                1
                - cache_result["input_frames"]
                / no_cache_result["input_frames"]
            )

            encoded_reduction = (
                1
                - cache_result["encoded_frames"]
                / no_cache_result["encoded_frames"]
            )

            speedup = (
                no_cache_result["time_mean"]
                / cache_result["time_mean"]
            )

            result = {
                "audio": audio_name,
                "lookahead": lookahead,
                "reference": reference,
                "cache": cache_result,
                "no_cache": no_cache_result,
                "speedup": speedup,
                "input_frame_reduction": input_reduction,
                "encoded_frame_reduction": encoded_reduction,
            }

            results.append(result)

            print()
            print(
                f"  Speedup: "
                f"{speedup:.2f}x"
            )

            print(
                f"  Input-frame reduction: "
                f"{input_reduction * 100:.2f}%"
            )

            print(
                f"  Encoded-frame reduction: "
                f"{encoded_reduction * 100:.2f}%"
            )

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        RESULTS_DIR / "benchmark_dataset.json"
    )

    with open(json_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------

    csv_path = (
        RESULTS_DIR / "benchmark_dataset.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "audio",
            "lookahead",
            "method",
            "wer_mean",
            "wer_std",
            "rtf_mean",
            "rtf_std",
            "time_mean_s",
            "time_std_s",
            "chunks",
            "input_frames",
            "encoded_frames",
            "speedup",
            "input_frame_reduction",
            "encoded_frame_reduction",
            "final_text",
        ])

        for result in results:

            for method_name in (
                "cache",
                "no_cache",
            ):

                method_result = result[
                    method_name
                ]

                writer.writerow([
                    result["audio"],
                    str(result["lookahead"]),
                    method_name,
                    method_result["wer_mean"],
                    method_result["wer_std"],
                    method_result["rtf_mean"],
                    method_result["rtf_std"],
                    method_result["time_mean"],
                    method_result["time_std"],
                    method_result["chunks"],
                    method_result["input_frames"],
                    method_result["encoded_frames"],
                    result["speedup"],
                    result["input_frame_reduction"],
                    result["encoded_frame_reduction"],
                    method_result["final_text"],
                ])

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print()
    print()
    print("=" * 130)
    print("FINAL SUMMARY")
    print("=" * 130)

    print()

    header = (
        f"{'Audio':<16}"
        f"{'Lookahead':<13}"
        f"{'Cache RTF':<12}"
        f"{'No-cache RTF':<14}"
        f"{'Speedup':<10}"
        f"{'WER':<8}"
        f"{'Input ↓':<10}"
        f"{'Encoded ↓':<12}"
    )

    print(header)
    print("-" * 130)

    for result in results:

        cache = result["cache"]
        no_cache = result["no_cache"]

        print(
            f"{result['audio']:<16}"
            f"{str(result['lookahead']):<13}"
            f"{cache['rtf_mean']:<12.4f}"
            f"{no_cache['rtf_mean']:<14.4f}"
            f"{result['speedup']:<10.2f}x"
            f"{cache['wer_mean']:<8.4f}"
            f"{result['input_frame_reduction'] * 100:<10.2f}%"
            f"{result['encoded_frame_reduction'] * 100:<12.2f}%"
        )

    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
