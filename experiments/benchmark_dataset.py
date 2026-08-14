from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import torch
from jiwer import (
    Compose,
    RemoveMultipleSpaces,
    RemovePunctuation,
    Strip,
    ToLowerCase,
    wer,
)

from cachespeech.audio import extract_features, load_audio
from cachespeech.model import CacheSpeechModel


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


NORMALIZE = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])


def synchronize(model: CacheSpeechModel) -> None:
    if model.device.startswith("cuda"):
        torch.cuda.synchronize()


def get_streaming_sizes(encoder):
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
    FastConformer streaming requires pre-encoder history
    to be present at the beginning of each streaming window.
    """

    return torch.nn.functional.pad(
        features,
        (pre_encode_cache, 0),
    )


def run_cached(
    model: CacheSpeechModel,
    features: torch.Tensor,
) -> dict:
    """
    Cache-aware streaming.

    Each streaming chunk is processed once and the encoder
    cache carries state between chunks.
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

    final_text = ""
    num_chunks = 0
    encoded_frames = 0
    input_frames = 0

    start = 0

    while start < features.shape[-1] - pre_encode_cache:

        end = min(
            start + window_size,
            features.shape[-1],
        )

        chunk = features[:, :, start:end]

        if chunk.shape[-1] < window_size:
            chunk = torch.nn.functional.pad(
                chunk,
                (0, window_size - chunk.shape[-1]),
            )

        input_frames += chunk.shape[-1]

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

        encoded_frames += encoded_len.item()
        num_chunks += 1

        final_text = partial_hypotheses[0].text

        start += shift_size

    return {
        "text": final_text,
        "chunks": num_chunks,
        "encoded_frames": encoded_frames,
        "input_frames": input_frames,
    }


def run_no_cache(
    model: CacheSpeechModel,
    features: torch.Tensor,
) -> dict:
    """
    No-cache streaming baseline.

    At every streaming step, the complete audio prefix observed
    so far is reprocessed from scratch using the encoder's normal
    forward() path.

    No encoder state is reused between steps.

    This intentionally performs redundant computation:

        prefix_1
        prefix_1 + prefix_2
        prefix_1 + prefix_2 + prefix_3
        ...

    The encoded frame count therefore represents the cumulative
    amount of encoder output generated across all recomputations.
    """

    encoder = model.encoder
    decoder = model.model.decoding

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

    final_text = ""
    num_chunks = 0
    encoded_frames = 0
    input_frames = 0

    start = 0

    while start < features.shape[-1] - pre_encode_cache:

        end = min(
            start + window_size,
            features.shape[-1],
        )

        # Reprocess the entire prefix observed so far.
        accumulated = features[:, :, :end]

        # Keep the same padding convention as the streaming
        # benchmark for the final incomplete window.
        if accumulated.shape[-1] < window_size:
            accumulated = torch.nn.functional.pad(
                accumulated,
                (0, window_size - accumulated.shape[-1]),
            )

        input_frames += accumulated.shape[-1]

        accumulated_length = torch.tensor(
            [accumulated.shape[-1]],
            device=model.device,
            dtype=torch.long,
        )

        with torch.inference_mode():

            # IMPORTANT:
            #
            # Do NOT use cache_aware_stream_step() here.
            #
            # The normal encoder forward() performs a fresh
            # encoder pass over the entire accumulated prefix.
            #
            # No encoder cache is supplied.
            encoded, encoded_len = encoder(
                audio_signal=accumulated,
                length=accumulated_length,
            )

            hypotheses = (
                decoder.rnnt_decoder_predictions_tensor(
                    encoded,
                    encoded_len,
                    return_hypotheses=True,
                    partial_hypotheses=None,
                )
            )

        # This is deliberately cumulative.
        #
        # If prefix lengths are:
        #
        #   70 + 140 + 210 + ...
        #
        # then all of those encoded outputs are counted because
        # the no-cache baseline recomputes all of them.
        encoded_frames += encoded_len.item()

        num_chunks += 1

        final_text = hypotheses[0].text

        start += shift_size

    return {
        "text": final_text,
        "chunks": num_chunks,
        "encoded_frames": encoded_frames,
        "input_frames": input_frames,
    }


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

        normalized_reference = NORMALIZE(reference)

        normalized_hypothesis = NORMALIZE(
            result["text"]
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
        "wer_mean": statistics.mean(wers),
        "wer_std": (
            statistics.stdev(wers)
            if len(wers) > 1
            else 0.0
        ),
        "time_mean": statistics.mean(times),
        "time_std": (
            statistics.stdev(times)
            if len(times) > 1
            else 0.0
        ),
        "rtf_mean": statistics.mean(rtfs),
        "rtf_std": (
            statistics.stdev(rtfs)
            if len(rtfs) > 1
            else 0.0
        ),
        "chunks": last_result["chunks"],
        "encoded_frames": last_result["encoded_frames"],
        "input_frames": last_result["input_frames"],
        "final_text": last_result["text"],
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
                run_cached,
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
