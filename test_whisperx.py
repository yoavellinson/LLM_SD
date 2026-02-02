import whisperx
import gc
from whisperx.diarize import DiarizationPipeline
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from hf_token import HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

device = "cuda"
audio_file = "/home/workspace/yoavellinson/LLM_SD/meeting_EN2006a_audio_8.wav"
batch_size = 1 
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

model = whisperx.load_model("large-v2", device, compute_type=compute_type)


audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size,language='en')
print(result["segments"]) # before alignment

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio,max_speakers=3)
result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs