import time
import torch

from cachespeech.audio import load_audio, extract_features
from cachespeech.model import CacheSpeechModel
from cachespeech.evaluation import run_no_cache


AUDIO_PATH = "audio/clean-1.wav"


def main():
    print("Loading model...")
    model = CacheSpeechModel()

    print()
    print("Loading audio...")
    waveform, sample_rate = load_audio(AUDIO_PATH)

    features, _ = extract_features(
        model,
        waveform,
        sample_rate,
    )

    print("Features:", features.shape)
    print()

    result = run_no_cache(
        model,
        features,
    )

    print()
    print("=" * 70)
    print("BASELINE")
    print("=" * 70)

    print()
    print("Final transcript:")
    print(repr(result["final_text"]))

    print()
    print("Chunks:", result["num_chunks"])
    print("Encoded frames:", result["encoded_frames"])
    print(
        "Inference time:",
        round(result["inference_time"], 4),
        "seconds",
    )


if __name__ == "__main__":
    main()
