from cachespeech.streaming import StreamingState

def __init__(self, model, lookahead: list[int] | None = None):
    self.model = model
    self.state = StreamingState()

    self.encoder = model.encoder

    if lookahead is not None:
        self.set_lookahead(lookahead)
    else:
        self.encoder.setup_streaming_params()

    self.drop_extra_pre_encoded = (
        self.encoder.streaming_cfg.drop_extra_pre_encoded
    )

    self._initialize_cache()