import time
import torch
from jiwer import wer

from cachespeech.audio import extract_features, load_audio
from cachespeech.model import CacheSpeechModel
from cachespeech.evaluation import (
    run_cache_aware,
    warm_up,
    stability_score,
    mean,
    std,
    WER_TRANSFORM,
)


AUDIO_PATH = "audio/clean-1.wav"

REFERENCE_TEXT = "Hello how are you how was your morning"

LOOKAHEADS = [
    [70, 13],
    [70, 6],
    [70, 1],
    [70, 0],
]

NUM_RUNS = 5


def run_single_evaluation(
    model,
    features,
    waveform,
    sample_rate,
    lookahead,
):
    """
    Run one measured streaming evaluation.
    """

    model.set_lookahead(lookahead)

    # ------------------------------------------------------------
    # Measured inference
    # ------------------------------------------------------------

    result = run_cache_aware(
        model,
        features,
    )

    final_text = result["final_text"]

    # ------------------------------------------------------------
    # WER
    # ------------------------------------------------------------

    normalized_reference = WER_TRANSFORM(
        REFERENCE_TEXT,
    )

    normalized_hypothesis = WER_TRANSFORM(
        final_text,
    )

    error_rate = wer(
        normalized_reference,
        normalized_hypothesis,
    )

    # ------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------

    stability = stability_score(
        result["transcripts"],
    )

    # ------------------------------------------------------------
    # RTF
    # ------------------------------------------------------------

    audio_duration = (
        waveform.shape[-1] / sample_rate
    )

    rtf = (
        result["inference_time"]
        / audio_duration
    )

    return {
        "wer": error_rate,
        "rtf": rtf,
        "inference_time": result["inference_time"],
        "chunks": result["num_chunks"],
        "encoded_frames": result["encoded_frames"],
        "stability": stability,
        "final_text": final_text,
    }


def main():

    # ============================================================
    # Load model once
    # ============================================================

    print("Loading model...")

    model = CacheSpeechModel()

    # ============================================================
    # Load audio once
    # ============================================================

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

    # ============================================================
    # Evaluation
    # ============================================================

    print()
    print("=" * 90)
    print("MULTI-RUN FIXED LOOKAHEAD EVALUATION")
    print("=" * 90)

    print()
    print(
        f"Runs per lookahead: {NUM_RUNS}"
    )

    all_results = []

    for lookahead in LOOKAHEADS:

        print()
        print("-" * 90)
        print(f"LOOKAHEAD: {lookahead}")
        print("-" * 90)

        # --------------------------------------------------------
        # Warm-up ONCE per lookahead
        # --------------------------------------------------------

        print(
            "Warm-up...",
            end=" ",
            flush=True,
        )

        warm_up(
            model,
            features,
            lookahead,
        )

        print("done")

        # --------------------------------------------------------
        # Measured runs
        # --------------------------------------------------------

        run_results = []

        for run_number in range(
            1,
            NUM_RUNS + 1,
        ):

            print(
                f"Run {run_number}/{NUM_RUNS}...",
                end=" ",
                flush=True,
            )

            result = run_single_evaluation(
                model=model,
                features=features,
                waveform=waveform,
                sample_rate=sample_rate,
                lookahead=lookahead,
            )

            run_results.append(result)

            print(
                f"RTF={result['rtf']:.4f} "
                f"time={result['inference_time']:.4f}s "
                f"WER={result['wer']:.4f}"
            )

        # --------------------------------------------------------
        # Aggregate results
        # --------------------------------------------------------

        wers = [
            result["wer"]
            for result in run_results
        ]

        rtfs = [
            result["rtf"]
            for result in run_results
        ]

        inference_times = [
            result["inference_time"]
            for result in run_results
        ]

        stabilities = [
            result["stability"]
            for result in run_results
        ]

        chunks = [
            result["chunks"]
            for result in run_results
        ]

        encoded_frames = [
            result["encoded_frames"]
            for result in run_results
        ]

        aggregated = {
            "lookahead": lookahead,

            "wer_mean": mean(wers),
            "wer_std": std(wers),

            "rtf_mean": mean(rtfs),
            "rtf_std": std(rtfs),

            "time_mean": mean(inference_times),
            "time_std": std(inference_times),

            "stability_mean": mean(stabilities),
            "stability_std": std(stabilities),

            "chunks": chunks[0],
            "encoded_frames": encoded_frames[0],

            "final_text": run_results[-1]["final_text"],
        }

        all_results.append(
            aggregated
        )

        # --------------------------------------------------------
        # Print aggregate
        # --------------------------------------------------------

        print()
        print("Aggregate:")

        print(
            f"  WER:        "
            f"{aggregated['wer_mean']:.4f} "
            f"± {aggregated['wer_std']:.4f}"
        )

        print(
            f"  RTF:        "
            f"{aggregated['rtf_mean']:.4f} "
            f"± {aggregated['rtf_std']:.4f}"
        )

        print(
            f"  Time:       "
            f"{aggregated['time_mean']:.4f}s "
            f"± {aggregated['time_std']:.4f}s"
        )

        print(
            f"  Stability:  "
            f"{aggregated['stability_mean']:.4f} "
            f"± {aggregated['stability_std']:.4f}"
        )

        print(
            f"  Chunks:     "
            f"{aggregated['chunks']}"
        )

        print(
            f"  Encoded:    "
            f"{aggregated['encoded_frames']}"
        )

        print(
            f"  Final:      "
            f"{aggregated['final_text']!r}"
        )

    # ============================================================
    # Final summary
    # ============================================================

    print()
    print("=" * 90)
    print("MULTI-RUN SUMMARY")
    print("=" * 90)

    print()

    print(
        f"{'Lookahead':<15}"
        f"{'WER':<18}"
        f"{'RTF':<18}"
        f"{'Time (s)':<22}"
        f"{'Stability':<20}"
        f"{'Chunks':<10}"
    )

    print("-" * 105)

    for result in all_results:

        wer_text = (
            f"{result['wer_mean']:.4f}"
            f" ± {result['wer_std']:.4f}"
        )

        rtf_text = (
            f"{result['rtf_mean']:.4f}"
            f" ± {result['rtf_std']:.4f}"
        )

        time_text = (
            f"{result['time_mean']:.4f}"
            f" ± {result['time_std']:.4f}"
        )

        stability_text = (
            f"{result['stability_mean']:.4f}"
            f" ± {result['stability_std']:.4f}"
        )

        print(
            f"{str(result['lookahead']):<15}"
            f"{wer_text:<18}"
            f"{rtf_text:<18}"
            f"{time_text:<22}"
            f"{stability_text:<20}"
            f"{result['chunks']:<10}"
        )


if __name__ == "__main__":
    main()
