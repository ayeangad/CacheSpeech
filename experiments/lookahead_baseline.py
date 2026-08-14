import time

import torch

from cachespeech.audio import load_audio, extract_features
from cachespeech.model import CacheSpeechModel
from cachespeech.streaming import CacheSpeechStream


AUDIO_PATH = "audio/audio-example-1.wav"

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

    stream = CacheSpeechStream(
        model,
        lookahead=lookahead,
    )

    cfg = model.encoder.streaming_cfg

    if isinstance(cfg.chunk_size, list):
        chunk_size = cfg.chunk_size[1]
    else:
        chunk_size = cfg.chunk_size

    if isinstance(cfg.shift_size, list):
        shift_size = cfg.shift_size[1]
    else:
        shift_size = cfg.shift_size

    if isinstance(cfg.pre_encode_cache_size, list):
        pre_encode = cfg.pre_encode_cache_size[1]
    else:
        pre_encode = cfg.pre_encode_cache_size

    input_chunk_size = chunk_size + pre_encode

    print()
    print("Streaming configuration:")
    print("  lookahead:", lookahead)
    print("  chunk size:", cfg.chunk_size)
    print("  shift size:", cfg.shift_size)
    print("  valid output length:", cfg.valid_out_len)
    print()

    transcripts = []

    import torch.nn.functional as F
    features = F.pad(features, (pre_encode, 0))

    start = 0

    if model.device == "cuda":
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    while start < features.shape[-1] - pre_encode:

        end = min(
            start + input_chunk_size,
            features.shape[-1],
        )

        chunk = features[:, :, start:end]

        if chunk.shape[-1] == 0:
            break

        actual_length = chunk.shape[-1]
        if actual_length < input_chunk_size:
            pad_amount = input_chunk_size - actual_length
            chunk = F.pad(chunk, (0, pad_amount))

        text = stream.step(
            chunk,
            feature_length=chunk.shape[-1],
        )

        transcripts.append(text)

        print(
            f"chunk {len(transcripts) - 1:02d} "
            f"| input={chunk.shape[-1]:3d} "
            f"| text={text!r}"
        )

        start += shift_size

    if model.device == "cuda":
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - inference_start

    return {
        "lookahead": lookahead,
        "final_text": transcripts[-1] if transcripts else "",
        "transcripts": transcripts,
        "inference_time": inference_time,
        "num_chunks": len(transcripts),
        "chunk_size": chunk_size,
        "shift_size": shift_size,
    }


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
