import time

import torch

from cachespeech.audio import load_audio, extract_features
from cachespeech.model import CacheSpeechModel


AUDIO_PATH = "audio/audio-example-1.wav"


def run_baseline(model, features):
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

    if isinstance(cfg.chunk_size, list):
        input_chunk_size = cfg.chunk_size[1]
    else:
        input_chunk_size = cfg.chunk_size

    if isinstance(cfg.shift_size, list):
        input_shift = cfg.shift_size[1]
    else:
        input_shift = cfg.shift_size

    print("Streaming configuration:")
    print("  attention context:", encoder.att_context_size)
    print("  chunk size:", cfg.chunk_size)
    print("  shift size:", cfg.shift_size)
    print("  valid output length:", cfg.valid_out_len)
    print()

    start = 0

    while start < features.shape[-1]:

        end = min(
            start + input_chunk_size,
            features.shape[-1],
        )

        chunk = features[:, :, start:end]

        actual_length = chunk.shape[-1]
        if actual_length == 0:
            break
            
        if actual_length < input_chunk_size:
            import torch.nn.functional as F
            pad_amount = input_chunk_size - actual_length
            chunk = F.pad(chunk, (0, pad_amount))

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

        print(
            f"chunk {len(transcripts) - 1:02d} "
            f"| input={chunk.shape[-1]:3d} "
            f"| encoded={encoded_len.item():2d} "
            f"| text={text!r}"
        )

        start += input_shift

    return {
        "final_text": transcripts[-1] if transcripts else "",
        "transcripts": transcripts,
        "inference_time": total_inference_time,
        "encoded_frames": total_encoded_frames,
        "num_chunks": len(transcripts),
    }


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

    result = run_baseline(
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
