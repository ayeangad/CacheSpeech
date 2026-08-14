from dataclasses import dataclass

import torch

from nemo.collections.asr.inference.utils.context_manager import CacheAwareContext
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


@dataclass
class StreamingState:
    """State carried between consecutive streaming inference steps."""

    context: CacheAwareContext | None = None
    hypotheses: list[Hypothesis] | None = None

    def reset(self) -> None:
        """Reset encoder and decoder streaming state."""
        self.context = None
        self.hypotheses = None


class CacheSpeechStream:
    """Cache-aware streaming encoder."""

    def __init__(self, model):
        self.model = model
        self.state = StreamingState()

        self.encoder = model.encoder
        self.encoder.setup_streaming_params()

        self.drop_extra_pre_encoded = (
            self.encoder.streaming_cfg.drop_extra_pre_encoded
        )

        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Create the initial encoder cache state."""

        (
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
        ) = self.encoder.get_initial_cache_state(
            batch_size=1,
            dtype=torch.float32,
            device=self.model.device,
        )

        self.state.context = CacheAwareContext(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

    def reset(self) -> None:
        """Reset the stream and create a fresh encoder cache."""

        self.state.reset()
        self._initialize_cache()

    def encoder_step(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process one feature chunk through the cache-aware encoder.

        Returns:
            encoded: Encoder representations.
            encoded_length: Number of valid encoder frames.
        """

        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape [B, F, T], got {features.shape}"
            )

        feature_length = torch.tensor(
            [features.shape[-1]],
            device=features.device,
            dtype=torch.long,
        )

        with torch.inference_mode():
            (
                encoded,
                encoded_length,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = self.encoder.cache_aware_stream_step(
                processed_signal=features,
                processed_signal_length=feature_length,
                cache_last_channel=self.state.context.cache_last_channel,
                cache_last_time=self.state.context.cache_last_time,
                cache_last_channel_len=self.state.context.cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=self.drop_extra_pre_encoded,
            )

        self.state.context = CacheAwareContext(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

        return encoded, encoded_length
