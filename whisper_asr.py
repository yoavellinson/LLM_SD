# must be FIRST
import os
os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
os.environ["MODELOPT_DISABLE"] = "1"

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperASRWrapper:
    def __init__(self, model_id="openai/whisper-large-v2",batch_size=6 ,device=None):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            batch_size=batch_size,
            generate_kwargs={
            "language": "en",
            "task": "transcribe",
        },
        )

    @torch.no_grad()
    def __call__(self, wav_files):
        out = self.pipe(wav_files)
        if isinstance(wav_files, list):
            return [o["text"] for o in out]
        return [out["text"]]

