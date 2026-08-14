from pathlib import Path

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


# ============================================================
# Configuration
# ============================================================

AUDIO_DIR = Path("audio")

LOOKAHEADS = [
    [70, 13],
    [70, 6],
    [70, 1],
    [70, 0],
]

NUM_RUNS = 5


DATASET = [
    {
        "name": "clean-1",
        "path": AUDIO_DIR / "clean-1.wav",
        "reference": (
            "hello how are you how was your morning"
        ),
    },
    {
        "name": "clean-2",
        "path": AUDIO_DIR / "clean-2.wav",
        "reference": (
            "hello im doing good my evening was amazing"
        ),
    },
    {
        "name": "noise-1",
        "path": AUDIO_DIR / "noise 2.wav",
        "reference": (
            "hello this is noisy example audio second"
        ),
    },
    {
        "name": "women-1",
        "path": None,
        "pattern": "women-1*.wav",
        "reference": (
            "hi i miss you youre really cute and "
            "youre really hot and i miss you so much "
            "and youre so sweet i miss you ok thats "
            "longer than oh oh shit ok ill stop"
        ),
    },
]


# ============================================================
# Utilities
# ============================================================

def resolve_audio_path(item):
    """
    Resolve an audio path.

    Most files have an explicit path. The women recording uses
    a glob because its filename is truncated in the UI screenshot.
    """

    if item.get("path") is not None:
        path = item["path"]

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        return path

    pattern = item["pattern"]

    matches = sorted(
        AUDIO_DIR.glob(pattern)
    )

    if not matches:
        raise FileNotFoundError(
            f"No audio file found matching: "
            f"{AUDIO_DIR / pattern}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files matched {pattern}: "
            f"{matches}"
        )

    return matches[0]


# ============================================================
# One measured evaluation
# ============================================================

def evaluate_once(
    model,
    features,
    waveform,
    sample_rate,
    reference,
    lookahead,
):
    """
    Run one measured evaluation.
    """

    model.set_lookahead(lookahead)

    result = run_cache_aware(
        model,
        features,
    )

    final_text = result["final_text"]

    # --------------------------------------------------------
    # WER
    # --------------------------------------------------------

    normalized_reference = WER_TRANSFORM(
        reference
    )

    normalized_hypothesis = WER_TRANSFORM(
        final_text
    )

    error_rate = wer(
        normalized_reference,
        normalized_hypothesis,
    )

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    stability = stability_score(
        result["transcripts"]
    )

    # --------------------------------------------------------
    # RTF
    # --------------------------------------------------------

    audio_duration = (
        waveform.shape[-1]
        / sample_rate
    )

    rtf = (
        result["inference_time"]
        / audio_duration
    )

    return {
        "wer": error_rate,
        "rtf": rtf,
        "inference_time": (
            result["inference_time"]
        ),
        "stability": stability,
        "chunks": result["num_chunks"],
        "encoded_frames": (
            result["encoded_frames"]
        ),
        "final_text": final_text,
    }


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("Loading model...")

    model = CacheSpeechModel()

    print()
    print("=" * 100)
    print("MULTI-AUDIO LOOKAHEAD EVALUATION")
    print("=" * 100)

    print()
    print(f"Audio files: {len(DATASET)}")
    print(f"Lookaheads:  {len(LOOKAHEADS)}")
    print(f"Runs each:   {NUM_RUNS}")

    all_results = []

    # ========================================================
    # Audio loop
    # ========================================================

    for audio_index, item in enumerate(DATASET, 1):

        path = resolve_audio_path(item)

        print()
        print("=" * 100)
        print(
            f"AUDIO {audio_index}/{len(DATASET)}: "
            f"{item['name']}"
        )
        print("=" * 100)

        print("File:", path)

        waveform, sample_rate = load_audio(
            str(path)
        )

        features, _ = extract_features(
            model,
            waveform,
            sample_rate,
        )

        duration = (
            waveform.shape[-1]
            / sample_rate
        )

        print(
            f"Duration: {duration:.3f}s"
        )

        print(
            f"Features: {tuple(features.shape)}"
        )

        print(
            f"Reference: "
            f"{item['reference']!r}"
        )

        # ----------------------------------------------------
        # Lookahead loop
        # ----------------------------------------------------

        for lookahead in LOOKAHEADS:

            print()
            print("-" * 100)
            print(
                f"LOOKAHEAD: {lookahead}"
            )
            print("-" * 100)

            # ------------------------------------------------
            # Warm-up once
            # ------------------------------------------------

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

            run_results = []

            # ------------------------------------------------
            # Measured runs
            # ------------------------------------------------

            for run_number in range(
                1,
                NUM_RUNS + 1,
            ):

                print(
                    f"Run "
                    f"{run_number}/{NUM_RUNS}...",
                    end=" ",
                    flush=True,
                )

                result = evaluate_once(
                    model=model,
                    features=features,
                    waveform=waveform,
                    sample_rate=sample_rate,
                    reference=item["reference"],
                    lookahead=lookahead,
                )

                run_results.append(
                    result
                )

                print(
                    f"RTF={result['rtf']:.4f} "
                    f"time={result['inference_time']:.4f}s "
                    f"WER={result['wer']:.4f} "
                    f"stability={result['stability']:.4f}"
                )

            # ------------------------------------------------
            # Aggregate
            # ------------------------------------------------

            aggregated = {
                "audio": item["name"],
                "path": str(path),
                "lookahead": lookahead,

                "wer_mean": mean([
                    r["wer"]
                    for r in run_results
                ]),

                "wer_std": std([
                    r["wer"]
                    for r in run_results
                ]),

                "rtf_mean": mean([
                    r["rtf"]
                    for r in run_results
                ]),

                "rtf_std": std([
                    r["rtf"]
                    for r in run_results
                ]),

                "time_mean": mean([
                    r["inference_time"]
                    for r in run_results
                ]),

                "time_std": std([
                    r["inference_time"]
                    for r in run_results
                ]),

                "stability_mean": mean([
                    r["stability"]
                    for r in run_results
                ]),

                "stability_std": std([
                    r["stability"]
                    for r in run_results
                ]),

                "chunks": run_results[0]["chunks"],
                "encoded_frames": (
                    run_results[0]["encoded_frames"]
                ),

                "final_text": (
                    run_results[-1]["final_text"]
                ),
            }

            all_results.append(
                aggregated
            )

            print()
            print("Aggregate:")
            print(
                f"  WER:       "
                f"{aggregated['wer_mean']:.4f} "
                f"± {aggregated['wer_std']:.4f}"
            )
            print(
                f"  RTF:       "
                f"{aggregated['rtf_mean']:.4f} "
                f"± {aggregated['rtf_std']:.4f}"
            )
            print(
                f"  Stability: "
                f"{aggregated['stability_mean']:.4f} "
                f"± {aggregated['stability_std']:.4f}"
            )
            print(
                f"  Final:     "
                f"{aggregated['final_text']!r}"
            )

    # ========================================================
    # Per-audio summary
    # ========================================================

    print()
    print("=" * 100)
    print("PER-AUDIO SUMMARY")
    print("=" * 100)

    print()

    print(
        f"{'Audio':<14}"
        f"{'Lookahead':<15}"
        f"{'WER':<18}"
        f"{'RTF':<18}"
        f"{'Stability':<20}"
    )

    print("-" * 90)

    for result in all_results:

        print(
            f"{result['audio']:<14}"
            f"{str(result['lookahead']):<15}"
            f"{result['wer_mean']:.4f}"
            f" ± {result['wer_std']:.4f}"
            f"    "
            f"{result['rtf_mean']:.4f}"
            f" ± {result['rtf_std']:.4f}"
            f"    "
            f"{result['stability_mean']:.4f}"
            f" ± {result['stability_std']:.4f}"
        )

    # ========================================================
    # Aggregate across all audio
    # ========================================================

    print()
    print("=" * 100)
    print("CROSS-AUDIO SUMMARY")
    print("=" * 100)

    cross_audio = []

    for lookahead in LOOKAHEADS:

        matching = [
            r
            for r in all_results
            if r["lookahead"] == lookahead
        ]

        aggregated = {
            "lookahead": lookahead,

            "wer_mean": mean([
                r["wer_mean"]
                for r in matching
            ]),

            "wer_std": std([
                r["wer_mean"]
                for r in matching
            ]),

            "rtf_mean": mean([
                r["rtf_mean"]
                for r in matching
            ]),

            "rtf_std": std([
                r["rtf_mean"]
                for r in matching
            ]),

            "stability_mean": mean([
                r["stability_mean"]
                for r in matching
            ]),

            "stability_std": std([
                r["stability_mean"]
                for r in matching
            ]),
        }

        cross_audio.append(
            aggregated
        )

    print()

    print(
        f"{'Lookahead':<15}"
        f"{'WER':<18}"
        f"{'RTF':<18}"
        f"{'Stability':<20}"
    )

    print("-" * 75)

    for result in cross_audio:

        print(
            f"{str(result['lookahead']):<15}"
            f"{result['wer_mean']:.4f}"
            f" ± {result['wer_std']:.4f}"
            f"    "
            f"{result['rtf_mean']:.4f}"
            f" ± {result['rtf_std']:.4f}"
            f"    "
            f"{result['stability_mean']:.4f}"
            f" ± {result['stability_std']:.4f}"
        )

    # ========================================================
    # Pareto frontier
    # ========================================================

    print()
    print("=" * 100)
    print("PARETO FRONTIER")
    print("=" * 100)

    # A configuration is dominated if another configuration
    # has:
    #
    #   <= RTF
    #   >= stability
    #   <= WER
    #
    # with at least one strict improvement.

    pareto = []

    for candidate in cross_audio:

        dominated = False

        for other in cross_audio:

            if other is candidate:
                continue

            no_worse = (
                other["rtf_mean"]
                <= candidate["rtf_mean"]
                and
                other["stability_mean"]
                >= candidate["stability_mean"]
                and
                other["wer_mean"]
                <= candidate["wer_mean"]
            )

            better = (
                other["rtf_mean"]
                < candidate["rtf_mean"]
                or
                other["stability_mean"]
                > candidate["stability_mean"]
                or
                other["wer_mean"]
                < candidate["wer_mean"]
            )

            if no_worse and better:
                dominated = True
                break

        if not dominated:
            pareto.append(candidate)

    print()

    print(
        f"{'Lookahead':<15}"
        f"{'WER':<18}"
        f"{'RTF':<18}"
        f"{'Stability':<20}"
    )

    print("-" * 75)

    for result in pareto:

        print(
            f"{str(result['lookahead']):<15}"
            f"{result['wer_mean']:.4f}"
            f" ± {result['wer_std']:.4f}"
            f"    "
            f"{result['rtf_mean']:.4f}"
            f" ± {result['rtf_std']:.4f}"
            f"    "
            f"{result['stability_mean']:.4f}"
            f" ± {result['stability_std']:.4f}"
        )


if __name__ == "__main__":
    main()
