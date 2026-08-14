
from __future__ import annotations

import time

import torch
from jiwer import (
    wer,
    Compose,
    ToLowerCase,
    RemovePunctuation,
    RemoveMultipleSpaces,
    Strip,
)

from cachespeech.audio import load_audio, extract_features
from cachespeech.model import CacheSpeechModel


AUDIO_PATH = "audio/clean-1.wav"

REFERENCE_TEXT = "Hello how are you how was your morning"

LOOKAHEAD = [70, 6]


NORMALIZE = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])


def get_streaming_sizes(encoder):
    """Return the streaming geometry used by FastConformer."""

    cfg = encoder.streaming_cfg

    if isinstance(cfg.chunk_size, list):
        chunk_size = cfg.chunk_size[-1]
    else:
        chunk_size = cfg.chunk_size

    if isinstance(cfg.shift_size, list):
        shift_size = cfg.shift_size[-1]
    else:
        shift_size = cfg.shift_size

    pre_encode_cache = cfg.pre_encode_cache_size

    if isinstance(pre_encode_cache, list):
        pre_encode_cache = pre_encode_cache[-1]

    return chunk_size, shift_size, pre_encode_cache


def prepare_features(
    features: torch.Tensor,
    pre_encode_cache: int,
) -> torch.Tensor:
    """
    Add the left-side pre-encoder history required by
    FastConformer cache-aware streaming.
    """

    return torch.nn.functional.pad(
        features,
        (pre_encode_cache, 0),
    )


def run_cached(
    model,
    features,
):
    """
    CacheSpeech.

    Each streaming chunk is processed incrementally.
    Encoder state is carried from one step to the next.
    """

    encoder = model.encoder
    decoder = model.model.decoding
    cfg = encoder.streaming_cfg

    (
        chunk_size,
        shift_size,
        pre_encode_cache,
    ) = get_streaming_sizes(encoder)

    features = prepare_features(
        features,
        pre_encode_cache,
    )

    window_size = chunk_size + pre_encode_cache

    (
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
    ) = encoder.get_initial_cache_state(
        batch_size=1,
        dtype=torch.float32,
        device=model.device,
    )

    partial_hypotheses = None

    total_encoded_frames = 0
    total_input_frames = 0
    num_chunks = 0

    final_text = ""

    start = 0

    while start < features.shape[-1] - pre_encode_cache:

        end = min(
            start + window_size,
            features.shape[-1],
        )

        chunk = features[:, :, start:end]

        actual_length = chunk.shape[-1]

        if actual_length < window_size:
            chunk = torch.nn.functional.pad(
                chunk,
                (0, window_size - actual_length),
            )

        chunk_length = torch.tensor(
            [chunk.shape[-1]],
            device=model.device,
            dtype=torch.long,
        )

        with torch.inference_mode():

            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = encoder.cache_aware_stream_step(
                processed_signal=chunk,
                processed_signal_length=chunk_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=cfg.drop_extra_pre_encoded,
            )

            partial_hypotheses = (
                decoder.rnnt_decoder_predictions_tensor(
                    encoded,
                    encoded_len,
                    return_hypotheses=True,
                    partial_hypotheses=partial_hypotheses,
                )
            )

        total_input_frames += chunk.shape[-1]
        total_encoded_frames += encoded_len.item()
        num_chunks += 1

        final_text = partial_hypotheses[0].text

        start += shift_size

    return {
        "text": final_text,
        "chunks": num_chunks,
        "encoded_frames": total_encoded_frames,
        "input_frames_processed": total_input_frames,
    }


def run_no_cache(
    model,
    features,
):
    """
    Naive no-cache baseline.

    At every streaming update, recompute the entire audio
    prefix observed so far.

    Example:

        step 1: encode [0 : t1]
        step 2: encode [0 : t2]
        step 3: encode [0 : t3]
        ...

    No encoder state is reused between steps.
    """

    encoder = model.encoder
    decoder = model.model.decoding
    cfg = encoder.streaming_cfg

    (
        chunk_size,
        shift_size,
        pre_encode_cache,
    ) = get_streaming_sizes(encoder)

    features = prepare_features(
        features,
        pre_encode_cache,
    )

    window_size = chunk_size + pre_encode_cache

    total_encoded_frames = 0
    total_input_frames = 0
    num_chunks = 0

    final_text = ""

    start = 0

    while start < features.shape[-1] - pre_encode_cache:

        end = min(
            start + window_size,
            features.shape[-1],
        )

        # Recompute EVERYTHING observed so far.
        accumulated = features[:, :, :end]

        actual_length = accumulated.shape[-1]

        if actual_length < window_size:
            accumulated = torch.nn.functional.pad(
                accumulated,
                (0, window_size - actual_length),
            )

        accumulated_length = torch.tensor(
            [accumulated.shape[-1]],
            device=model.device,
            dtype=torch.long,
        )

        # Fresh cache/state for every invocation.
        (
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
        ) = encoder.get_initial_cache_state(
            batch_size=1,
            dtype=torch.float32,
            device=model.device,
        )

        with torch.inference_mode():

            (
                encoded,
                encoded_len,
                _,
                _,
                _,
            ) = encoder.cache_aware_stream_step(
                processed_signal=accumulated,
                processed_signal_length=accumulated_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=cfg.drop_extra_pre_encoded,
            )

            hypotheses = (
                decoder.rnnt_decoder_predictions_tensor(
                    encoded,
                    encoded_len,
                    return_hypotheses=True,
                    partial_hypotheses=None,
                )
            )

        total_input_frames += accumulated.shape[-1]
        total_encoded_frames += encoded_len.item()
        num_chunks += 1

        final_text = hypotheses[0].text

        start += shift_size

    return {
        "text": final_text,
        "chunks": num_chunks,
        "encoded_frames": total_encoded_frames,
        "input_frames_processed": total_input_frames,
    }


def benchmark_method(
    method,
    model,
    features,
    audio_duration,
):
    """Benchmark one method."""

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
        NORMALIZE(REFERENCE_TEXT),
        NORMALIZE(result["text"]),
    )

    return result


def print_result(
    name,
    result,
):
    print()
    print("-" * 65)
    print(name)
    print("-" * 65)

    print(
        "Final:",
        repr(result["text"]),
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
    print("=" * 65)
    print("CACHE VS NO-CACHE")
    print("=" * 65)

    print()
    print("Lookahead:", LOOKAHEAD)

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    print()
    print("Warming up...")

    _ = benchmark_method(
        run_cached,
        model,
        features,
        audio_duration,
    )

    _ = benchmark_method(
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
        run_cached,
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
    print("=" * 65)
    print("RESULT")
    print("=" * 65)

    print()

    print(
        f"{'Metric':<30}"
        f"{'CacheSpeech':<18}"
        f"{'No-cache':<18}"
    )

    print("-" * 65)

    print(
        f"{'WER':<30}"
        f"{cached['wer']:<18.4f}"
        f"{no_cache['wer']:<18.4f}"
    )

    print(
        f"{'Inference time (s)':<30}"
        f"{cached['inference_time']:<18.4f}"
        f"{no_cache['inference_time']:<18.4f}"
    )

    print(
        f"{'RTF':<30}"
        f"{cached['rtf']:<18.4f}"
        f"{no_cache['rtf']:<18.4f}"
    )

    print(
        f"{'Encoded frames':<30}"
        f"{cached['encoded_frames']:<18}"
        f"{no_cache['encoded_frames']:<18}"
    )

    print(
        f"{'Input frames processed':<30}"
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

    if cached["input_frames_processed"] > 0:

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