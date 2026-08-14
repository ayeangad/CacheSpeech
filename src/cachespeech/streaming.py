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
    """Cache-aware streaming encoder and RNNT decoder."""

    def __init__(
        self,
        model,
        lookahead: list[int] | None = None,
    ):
        self.model = model
        self.state = StreamingState()

        self.encoder = model.encoder

        if lookahead is not None:
            if lookahead not in self.model.lookahead_options:
                raise ValueError(
                    f"Unsupported lookahead {lookahead}. "
                    f"Supported options: {self.model.lookahead_options}"
                )

            self.encoder.set_default_att_context_size(
                att_context_size=lookahead,
            )
        else:
            self.encoder.setup_streaming_params()

        self.drop_extra_pre_encoded = (
            self.encoder.streaming_cfg.drop_extra_pre_encoded
        )

        self._initialize_cache()


    def set_lookahead(
        self,
        lookahead: list[int],
    ) -> None:
        """
        Switch the encoder to one of the supported attention lookaheads.

        Changing lookahead changes the encoder's streaming geometry,
        so the existing streaming cache must be discarded.
        """

        if lookahead not in self.model.lookahead_options:
            raise ValueError(
                f"Unsupported lookahead {lookahead}. "
                f"Supported options: {self.model.lookahead_options}"
            )

        self.encoder.set_default_att_context_size(
            att_context_size=lookahead,
        )

        self.drop_extra_pre_encoded = (
            self.encoder.streaming_cfg.drop_extra_pre_encoded
        )

        self.reset()

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

        Args:
            features: Mel features with shape [B, F, T].

        Returns:
            encoded: Encoder representations.
            encoded_length: Number of valid encoder frames.
        """

        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape [B, F, T], "
                f"got {features.shape}"
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

    def step(
        self,
        features: torch.Tensor,
        feature_length: int | None = None,
    ) -> str:
        """
        Process one streaming feature chunk and decode its RNNT hypothesis.

        Args:
            features: Mel features with shape [B, F, T].
            feature_length: Number of valid feature frames.

        Returns:
            Current best partial transcript.
        """

        if feature_length is None:
            feature_length = features.shape[-1]

        if feature_length <= 0:
            return ""

        encoded, encoded_length = self.encoder_step(
            features,
        )

        decoder = self.model.model.decoding

        with torch.inference_mode():
            self.state.hypotheses = (
                decoder.rnnt_decoder_predictions_tensor(
                    encoded,
                    encoded_length,
                    return_hypotheses=True,
                    partial_hypotheses=self.state.hypotheses,
                )
            )

        if not self.state.hypotheses:
            return ""

        return self.state.hypotheses[0].text
