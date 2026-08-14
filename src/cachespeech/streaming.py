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
    """
    Stateful cache-aware streaming ASR engine.

    Each call to `step()` processes one acoustic feature chunk and
    carries the encoder cache + RNNT hypothesis into the next step.
    """

    def __init__(self, model):
        self.model = model
        self.state = StreamingState()

    def step(
        self,
        features: torch.Tensor,
        feature_length: int | None = None,
    ) -> str:
        """
        Process one chunk of acoustic features.

        Args:
            features:
                Tensor shaped [B, 80, T].

            feature_length:
                Number of valid frames in the chunk.

        Returns:
            Current transcription hypothesis.
        """

        if features.ndim != 3:
            raise ValueError(
                f"Expected features with shape [B, 80, T], "
                f"got {tuple(features.shape)}"
            )

        if features.shape[1] != 80:
            raise ValueError(
                f"Expected 80 mel features, got {features.shape[1]}"
            )

        if feature_length is None:
            feature_length = features.shape[-1]

        length = torch.tensor(
            [feature_length],
            device=features.device,
            dtype=torch.long,
        )

        hypotheses, context = self.model.streaming.stream_step(
            processed_signal=features,
            processed_signal_length=length,
            context=self.state.context,
            previous_hypotheses=self.state.hypotheses,
        )

        self.state.context = context
        self.state.hypotheses = hypotheses

        if not hypotheses:
            return ""

        return hypotheses[0].text

    def reset(self) -> None:
        """Start a new utterance."""
        self.state.reset()
