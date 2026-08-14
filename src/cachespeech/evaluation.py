import statistics
import time

import torch
import torch.nn.functional as F
from jiwer import (
    Compose,
    RemoveMultipleSpaces,
    RemovePunctuation,
    Strip,
    ToLowerCase,
)


WER_TRANSFORM = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def std(values: list[float]) -> float:
    """Sample standard deviation."""
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def stability_score(transcripts: list[str]) -> float:
    """
    Fraction of consecutive transcript updates that remain unchanged.
    Higher = more stable partial transcript.
    """
    if len(transcripts) <= 1:
        return 1.0

    unchanged = sum(
        a == b
        for a, b in zip(
            transcripts,
            transcripts[1:],
        )
    )

    return unchanged / (len(transcripts) - 1)


def get_streaming_sizes(encoder) -> tuple[int, int, int]:
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
    FastConformer streaming requires pre-encoder history
    to be present at the beginning of each streaming window.
    """
    return F.pad(
        features,
        (pre_encode_cache, 0),
    )


def run_cache_aware(
    model,
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

    padded_features = prepare_features(
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
    transcripts = []

    total_inference_time = 0.0
    total_encoded_frames = 0
    total_input_frames = 0

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

        if actual_length < window_size:
            chunk = F.pad(
                chunk,
                (0, window_size - actual_length),
            )

        total_input_frames += chunk.shape[-1]

        chunk_length = torch.tensor(
            [chunk.shape[-1]],
            device=model.device,
            dtype=torch.long,
        )

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

        start += shift_size

    return {
        "text": transcripts[-1] if transcripts else "",
        "final_text": transcripts[-1] if transcripts else "",
        "transcripts": transcripts,
        "inference_time": total_inference_time,
        "encoded_frames": total_encoded_frames,
        "input_frames_processed": total_input_frames,  # For backward compatibility
        "input_frames": total_input_frames,
        "chunks": len(transcripts),
        "num_chunks": len(transcripts),
    }


def run_no_cache(
    model,
    features: torch.Tensor,
) -> dict:
    """
    No-cache streaming baseline.

    At every streaming step, the complete audio prefix observed
    so far is reprocessed from scratch using the encoder's normal
    forward() path.

    No encoder state is reused between steps.
    """
    encoder = model.encoder
    decoder = model.model.decoding

    (
        chunk_size,
        shift_size,
        pre_encode_cache,
    ) = get_streaming_sizes(encoder)

    padded_features = prepare_features(
        features,
        pre_encode_cache,
    )

    window_size = chunk_size + pre_encode_cache

    transcripts = []
    total_inference_time = 0.0
    total_encoded_frames = 0
    total_input_frames = 0

    start = 0

    while start < padded_features.shape[-1] - pre_encode_cache:

        end = min(
            start + window_size,
            padded_features.shape[-1],
        )

        # Reprocess the entire prefix observed so far.
        accumulated = padded_features[:, :, :end]

        actual_length = accumulated.shape[-1]
        
        if actual_length == 0:
            break

        # Keep the same padding convention as the streaming
        # benchmark for the final incomplete window.
        if actual_length < window_size:
            accumulated = F.pad(
                accumulated,
                (0, window_size - actual_length),
            )

        total_input_frames += accumulated.shape[-1]

        accumulated_length = torch.tensor(
            [accumulated.shape[-1]],
            device=model.device,
            dtype=torch.long,
        )

        if model.device.startswith("cuda"):
            torch.cuda.synchronize()

        inference_start = time.perf_counter()

        with torch.inference_mode():
            # IMPORTANT:
            # The normal encoder forward() performs a fresh
            # encoder pass over the entire accumulated prefix.
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

        if model.device.startswith("cuda"):
            torch.cuda.synchronize()

        total_inference_time += (
            time.perf_counter() - inference_start
        )

        total_encoded_frames += encoded_len.item()
        
        text = hypotheses[0].text
        transcripts.append(text)

        start += shift_size

    return {
        "text": transcripts[-1] if transcripts else "",
        "final_text": transcripts[-1] if transcripts else "",
        "transcripts": transcripts,
        "inference_time": total_inference_time,
        "encoded_frames": total_encoded_frames,
        "input_frames_processed": total_input_frames,  # For backward compatibility
        "input_frames": total_input_frames,
        "chunks": len(transcripts),
        "num_chunks": len(transcripts),
    }


def warm_up(model, features: torch.Tensor, lookahead: list[int]) -> None:
    """
    Perform one unmeasured inference pass to warm up the GPU/model.
    """
    model.set_lookahead(lookahead)

    # Clear stale CUDA synchronization state before warm-up.
    if model.device.startswith("cuda"):
        torch.cuda.synchronize()

    run_cache_aware(model, features)

    if model.device.startswith("cuda"):
        torch.cuda.synchronize()
