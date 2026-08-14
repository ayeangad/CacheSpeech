import statistics
import time

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


AUDIO_PATH = "audio/audio-example-1.wav"

REFERENCE_TEXT = "Hello how are you how was your morning"

LOOKAHEADS = [
    [70, 13],
    [70, 6],
    [70, 1],
    [70, 0],
]

NUM_RUNS = 5


WER_TRANSFORM = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])


def run_stream(model, features):
    """
    Run one complete cache-aware streaming inference pass.
    """

    encoder = model.encoder
    decoder = model.model.decoding
    cfg = encoder.streaming_cfg

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
    transcripts = []

    total_inference_time = 0.0
    total_encoded_frames = 0

    # ------------------------------------------------------------
    # Streaming geometry
    # ------------------------------------------------------------

    if isinstance(cfg.chunk_size, list):
        input_chunk_size = cfg.chunk_size[-1]
    else:
        input_chunk_size = cfg.chunk_size

    if isinstance(cfg.shift_size, list):
        input_shift = cfg.shift_size[-1]
    else:
        input_shift = cfg.shift_size

    # FastConformer requires pre-encoder history
    # on every streaming chunk.
    pre_encode_cache = cfg.pre_encode_cache_size

    if isinstance(pre_encode_cache, list):
        pre_encode_cache = pre_encode_cache[-1]

    # Actual input window = streaming chunk + pre-encoder history.
    window_size = input_chunk_size + pre_encode_cache

    # Add the initial historical context.
    padded_features = torch.nn.functional.pad(
        features,
        (pre_encode_cache, 0),
    )

    # ------------------------------------------------------------
    # Streaming loop
    # ------------------------------------------------------------

    start = 0

    while start < padded_features.shape[-1] - pre_encode_cache:

        end = min(
            start + window_size,
            padded_features.shape[-1],
        )

        chunk = padded_features[:, :, start:end]

        actual_length = chunk.shape[-1]

        if actual_length == 0:
            break

        # Pad the final chunk if necessary.
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

        # Synchronize so GPU timing is accurate.
        if model.device.startswith("cuda"):
            torch.cuda.synchronize()

        inference_start = time.perf_counter()

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

        if model.device.startswith("cuda"):
            torch.cuda.synchronize()

        total_inference_time += (
            time.perf_counter() - inference_start
        )

        total_encoded_frames += encoded_len.item()

        text = partial_hypotheses[0].text

        transcripts.append(text)

        start += input_shift

    return {
        "final_text": transcripts[-1] if transcripts else "",
        "transcripts": transcripts,
        "inference_time": total_inference_time,
        "encoded_frames": total_encoded_frames,
        "num_chunks": len(transcripts),
    }


def warm_up(model, features, lookahead):
    """
    Perform one unmeasured inference pass to warm up the GPU/model.

    This is intentionally separate from run_single_evaluation()
    so the warm-up cannot recursively call itself.
    """

    model.set_lookahead(lookahead)

    # Clear stale CUDA synchronization state before warm-up.
    if model.device.startswith("cuda"):
        torch.cuda.synchronize()

    run_stream(
        model,
        features,
    )

    if model.device.startswith("cuda"):
        torch.cuda.synchronize()


def stability_score(transcripts):
    """
    Fraction of consecutive transcript updates
    that remained unchanged.

    Higher = more stable transcript.
    """

    if len(transcripts) <= 1:
        return 1.0

    unchanged = sum(
        a == b
        for a, b in zip(transcripts, transcripts[1:])
    )

    return unchanged / (len(transcripts) - 1)


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

    result = run_stream(
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


def mean(values):
    return statistics.mean(values)


def std(values):
    """
    Sample standard deviation.

    With 5 runs this gives us an estimate of
    run-to-run variability.
    """

    if len(values) <= 1:
        return 0.0

    return statistics.stdev(values)


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
