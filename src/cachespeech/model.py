import torch
import nemo.collections.asr as nemo_asr


MODEL_NAME = "stt_en_fastconformer_hybrid_large_streaming_multi"

SUPPORTED_LOOKAHEADS = [
    [70, 13],
    [70, 6],
    [70, 1],
    [70, 0],
]


class CacheSpeechModel:
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Loading {MODEL_NAME}...")
        print(f"Device: {self.device}")

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=MODEL_NAME,
            map_location=self.device,
        )

        self.model.eval()

        print("Model loaded.")

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def preprocessor(self):
        return self.model.preprocessor

    @property
    def lookahead_options(self) -> list[list[int]]:
        return self.encoder.att_context_size_all

    @property
    def default_lookahead(self) -> list[int]:
        return self.encoder.att_context_size

    @property
    def subsampling_factor(self) -> int:
        return self.encoder.subsampling_factor

