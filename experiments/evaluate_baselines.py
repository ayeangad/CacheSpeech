
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


AUDIO_PATH = "audio/audio-example-1.wav"

# Use the exact words spoken in the audio.
REFERENCE_TEXT = "Hello how are you how was your morning"

LOOKAHEADS = [
    [70, 13],
    [70, 6],
    [70, 1],
    [70, 0],
]


WER_TRANSFORM = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])


def run_stream(model, features):
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

    # FastConformer requires historical feature frames
    # for the convolutional pre-encoder.
    pre_encode_cache = cfg.pre_encode_cache_size

    if isinstance(pre_encode_cache, list):
        pre_encode_cache = pre_encode_cache[-1]

    # The actual input window consists of:
    #
    #   previous pre-encoder context
    #   +
    #   current streaming chunk
    #
    window_size = input_chunk_size + pre_encode_cache

    # Add initial left-side context.
    padded_features = torch.nn.functional.pad(
        features,
        (pre_encode_cache, 0),
    )

    # ------------------------------------------------------------
    # Streaming inference
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

        # Final chunk may be smaller than the required
        # convolutional input window.
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

        # Synchronize before timing CUDA work.
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

        # Synchronize after CUDA work so timing includes GPU execution.
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


def stability_score(transcripts):
    """
    Fraction of consecutive transcript updates that
    remained unchanged.

    Higher = more stable transcript.
    """

    if len(transcripts) <= 1:
        return 1.0

    unchanged = sum(
        a == b
        for a, b in zip(transcripts, transcripts[1:])
    )

    return unchanged / (len(transcripts) - 1)


def main():

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    print("Loading model...")
    model = CacheSpeechModel()

    # ------------------------------------------------------------
    # Load audio
    # ------------------------------------------------------------

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

    print("Features:", features.shape)

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("FIXED LOOKAHEAD EVALUATION")
    print("=" * 80)

    results = []

    for lookahead in LOOKAHEADS:

        print()
        print("-" * 80)
        print(f"LOOKAHEAD: {lookahead}")
        print("-" * 80)

        # Change FastConformer attention context.
        model.set_lookahead(
            lookahead,
        )

        result = run_stream(
            model,
            features,
        )

        final_text = result["final_text"]

        # --------------------------------------------------------
        # WER
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Stability
        # --------------------------------------------------------

        stability = stability_score(
            result["transcripts"],
        )

        # --------------------------------------------------------
        # Real-time factor
        # --------------------------------------------------------

        audio_duration = (
            waveform.shape[-1] / sample_rate
        )

        rtf = (
            result["inference_time"]
            / audio_duration
        )

        # --------------------------------------------------------
        # Store result
        # --------------------------------------------------------

        results.append(
            {
                "lookahead": lookahead,
                "wer": error_rate,
                "rtf": rtf,
                "inference_time": result["inference_time"],
                "chunks": result["num_chunks"],
                "encoded_frames": result["encoded_frames"],
                "stability": stability,
                "final_text": final_text,
            }
        )

        # --------------------------------------------------------
        # Print result
        # --------------------------------------------------------

        print()
        print("Reference:", repr(REFERENCE_TEXT))
        print("Final:", repr(final_text))

        print()
        print("Normalized reference:")
        print(repr(normalized_reference))

        print("Normalized hypothesis:")
        print(repr(normalized_hypothesis))

        print()
        print("WER:", round(error_rate, 4))
        print("RTF:", round(rtf, 4))

        print(
            "Inference:",
            round(
                result["inference_time"],
                4,
            ),
            "s",
        )

        print(
            "Chunks:",
            result["num_chunks"],
        )

        print(
            "Encoded frames:",
            result["encoded_frames"],
        )

        print(
            "Stability:",
            round(
                stability,
                4,
            ),
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print()

    print(
        f"{'Lookahead':<15}"
        f"{'WER':<10}"
        f"{'RTF':<10}"
        f"{'Time':<12}"
        f"{'Chunks':<10}"
        f"{'Stability':<12}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{str(result['lookahead']):<15}"
            f"{result['wer']:<10.4f}"
            f"{result['rtf']:<10.4f}"
            f"{result['inference_time']:<12.4f}"
            f"{result['chunks']:<10}"
            f"{result['stability']:<12.4f}"
        )


if __name__ == "__main__":
    main()