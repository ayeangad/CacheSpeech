import time
import torch

from cachespeech.audio import load_audio, extract_features
from cachespeech.model import CacheSpeechModel
from cachespeech.evaluation import run_cache_aware, get_streaming_sizes


AUDIO_PATH = "audio/clean-1.wav"

LOOKAHEADS = [
    [70, 13],
    [70, 6],
    [70, 1],
    [70, 0],
]


def run_stream(
    model: CacheSpeechModel,
    features: torch.Tensor,
    lookahead: list[int],
) -> dict:
    """Run cache-aware streaming inference for one lookahead."""

    model.set_lookahead(lookahead)

    chunk_size, shift_size, pre_encode_cache = get_streaming_sizes(model.encoder)

    print()
    print("Streaming configuration:")
    print("  lookahead:", lookahead)
    print("  chunk size:", chunk_size)
    print("  shift size:", shift_size)
    print()

    # The centralized function returns everything we need
    result = run_cache_aware(model, features)

    result["lookahead"] = lookahead
    result["chunk_size"] = chunk_size
    result["shift_size"] = shift_size

    return result


def main():
    print("Loading model...")
    model = CacheSpeechModel()

    print()
    print("Loading audio...")

    waveform, sample_rate = load_audio(
        AUDIO_PATH,
    )

    features, feature_lengths = extract_features(
        model,
        waveform,
        sample_rate,
    )

    valid_frames = feature_lengths.item()

    print("Features:", features.shape)
    print("Valid feature frames:", valid_frames)

    # Only process valid feature frames.
    features = features[:, :, :valid_frames]

    results = []

    for lookahead in LOOKAHEADS:

        print()
        print("=" * 70)
        print(f"LOOKAHEAD: {lookahead}")
        print("=" * 70)

        result = run_stream(
            model,
            features,
            lookahead,
        )

        results.append(result)

        print()
        print("Final:")
        print(repr(result["final_text"]))

        print()
        print(
            "Inference time:",
            round(result["inference_time"], 4),
            "seconds",
        )

        print(
            "Chunks:",
            result["num_chunks"],
        )

    print()
    print("=" * 70)
    print("LOOKAHEAD SUMMARY")
    print("=" * 70)

    for result in results:

        print()
        print("Lookahead:", result["lookahead"])
        print("  chunk size:", result["chunk_size"])
        print("  shift size:", result["shift_size"])
        print("  chunks:", result["num_chunks"])
        print(
            "  inference:",
            round(result["inference_time"], 4),
            "s",
        )
        print(
            "  final:",
            repr(result["final_text"]),
        )


if __name__ == "__main__":
    main()
