from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf


TARGET_SAMPLE_RATE = 16_000


def load_audio(
    path: str | Path,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """
    Load a mono audio file and resample it to the target sample rate.

    Returns:
        waveform: Tensor with shape [1, samples].
        sample_rate: The resulting sample rate.
    """

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
    )

    waveform = torch.from_numpy(audio)

    # Convert mono audio to [1, samples].
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        # soundfile returns [samples, channels].
        waveform = waveform.mean(dim=1, keepdim=True).transpose(0, 1)
    else:
        raise ValueError(
            f"Expected 1D or 2D audio, got shape {waveform.shape}"
        )

    if sample_rate != target_sample_rate:
        new_length = round(
            waveform.shape[-1]
            * target_sample_rate
            / sample_rate
        )

        waveform = F.interpolate(
            waveform.unsqueeze(0),
            size=new_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

        sample_rate = target_sample_rate

    return waveform, sample_rate


def extract_features(
    model,
    waveform: torch.Tensor,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a waveform into NeMo mel-spectrogram features.
    """

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_RATE} Hz audio, got {sample_rate} Hz"
        )

    waveform = waveform.to(model.device)

    length = torch.tensor(
        [waveform.shape[-1]],
        device=model.device,
        dtype=torch.long,
    )

    features, feature_lengths = model.preprocessor(
        input_signal=waveform,
        length=length,
    )

    return features, feature_lengths


def chunk_features(
    features: torch.Tensor,
    chunk_size: list[int] | int,
    shift_size: list[int] | int,
) -> list[torch.Tensor]:
    """
    Split feature frames into streaming chunks.

    For the FastConformer model, chunk_size and shift_size are
    provided as [left, right] values because the model's
    depthwise-striding pre-encoder has asymmetric streaming
    requirements.
    """

    if features.dim() != 3:
        raise ValueError(
            f"Expected features with shape [B, F, T], got {features.shape}"
        )

    if isinstance(chunk_size, int):
        chunk = chunk_size
    else:
        chunk = max(chunk_size)

    if isinstance(shift_size, int):
        shift = shift_size
    else:
        shift = max(shift_size)

    chunks = []

    start = 0
    total_frames = features.shape[-1]

    while start < total_frames:
        end = min(start + chunk, total_frames)

        chunks.append(
            features[:, :, start:end]
        )

        if end >= total_frames:
            break

        start += shift

    return chunks

