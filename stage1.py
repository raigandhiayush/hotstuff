!nvidia-smi
!pip install -q transformers datasets accelerate librosa soundfile evaluate jiwer

!pip uninstall -y torch torchvision torchaudio
!pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q -U peft
import torch
print(torch.__version__)

!pip install pyannote.audio


!pip install --upgrade scipy

# Restart the runtime programmatically to load the newly installed numpy and scipy
import os
os.kill(os.getpid(), 9)

import sys
# Prioritize custom installed libraries over the broken system path
if '/usr/local/lib/python3.13/dist-packages' in sys.path:
    sys.path.remove('/usr/local/lib/python3.13/dist-packages')
sys.path.insert(0, '/usr/local/lib/python3.13/dist-packages')

import os
import torch
import soundfile as sf
import numpy as np

from jiwer import wer
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from pyannote.audio import Pipeline

# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "openai/whisper-small"

# ============================================================
# PATH CONFIGURATION
# ============================================================

import os

# LoRA model directory in the CURRENT WORKING DIRECTORY
BACKUP_DIR = "./whisper-small-lora-backup"

# Audio file path
# It is assumed that audio file is .wav
# Else it is converted to .wav from main script
AUDIO_PATH = "./meeting.wav"


# ------------------------------------------------------------
# Validate model directory
# ------------------------------------------------------------

if not os.path.isdir(BACKUP_DIR):
    raise FileNotFoundError(
        f"LoRA model directory not found:\n"
        f"  {os.path.abspath(BACKUP_DIR)}"
    )

print(f"LoRA model found: {os.path.abspath(BACKUP_DIR)}")


# ------------------------------------------------------------
# Validate audio file
# ------------------------------------------------------------

if not os.path.isfile(AUDIO_PATH):
    raise FileNotFoundError(
        f"Audio file not found:\n"
        f"  {os.path.abspath(AUDIO_PATH)}"
    )

print(f"Audio file found: {os.path.abspath(AUDIO_PATH)}")

audio_base, _ = os.path.splitext(AUDIO_PATH)

OUTPUT_PATH = f"{audio_base}_raw_transcript.txt"

print(f"Output transcript: {os.path.abspath(OUTPUT_PATH)}")



MAX_CHUNK_SECONDS = 30

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("Meeting Transcription Pipeline")
print("=" * 70)

print(f"Device: {device}")
print(f"Audio:  {AUDIO_PATH}")


# ============================================================
# 2. LOAD WHISPER PROCESSOR + LoRA MODEL
# ============================================================

print("\nLoading Whisper processor...")

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME
)

print("Loading base Whisper model...")

base_model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME
).to(device)

base_model.generation_config.language = "english"
base_model.generation_config.task = "transcribe"

# Fix PEFT compatibility issue with transformers generation kwargs
if not hasattr(base_model, "_prepare_encoder_decoder_kwargs_for_generation"):
    base_model._prepare_encoder_decoder_kwargs_for_generation = lambda *args, **kwargs: {}

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    BACKUP_DIR
)

model.eval()

print("LoRA model loaded successfully.")


# ============================================================
# 3. LOAD AUDIO FROM WORKING DIRECTORY
# ============================================================

if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(
        f"Audio file not found: {AUDIO_PATH}\n"
        f"Current working directory: {os.getcwd()}"
    )

print("\nLoading audio...")

audio, original_sr = sf.read(AUDIO_PATH)

# Stereo -> mono
if audio.ndim > 1:
    audio = audio.mean(axis=1)

audio = audio.astype(np.float32)

print(f"Original sampling rate: {original_sr}")
print(
    f"Original duration: "
    f"{len(audio) / original_sr / 60:.2f} minutes"
)


# ============================================================
# 4. RESAMPLE AUDIO FOR WHISPER
# ============================================================

TARGET_SR = 16000

if original_sr != TARGET_SR:

    print(
        f"Resampling audio: "
        f"{original_sr} Hz -> {TARGET_SR} Hz"
    )

    from scipy.signal import resample_poly

    audio = resample_poly(
        audio,
        TARGET_SR,
        original_sr
    ).astype(np.float32)

    sr = TARGET_SR

else:
    sr = original_sr

print(f"Whisper sampling rate: {sr}")
print(
    f"Whisper audio duration: "
    f"{len(audio) / sr / 60:.2f} minutes"
)


# ============================================================
# 5. LOAD PYANNOTE DIARIZATION PIPELINE
# ============================================================

print("\nLoading pyannote diarization model...")

HF_TOKEN = "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"

diarization_pipeline = None
diarization_available = False

try:

    if HF_TOKEN is None:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set."
        )

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN
    )

    if diarization_pipeline is None:
        raise RuntimeError(
            "Hugging Face returned no diarization pipeline."
        )

    diarization_pipeline.to(
        torch.device(device)
    )

    diarization_available = True

    print("Pyannote diarization model loaded successfully.")


except Exception as e:

    print("\n" + "=" * 70)
    print("WARNING: SPEAKER DIARIZATION UNAVAILABLE")
    print("=" * 70)

    error_message = str(e)

    if (
        "GatedRepoError" in error_message
        or "403" in error_message
        or "restricted" in error_message
        or "authorized list" in error_message
    ):
        print(
            "The Hugging Face model "
            "'pyannote/speaker-diarization-community-1' "
            "is gated."
        )
        print(
            "Your Hugging Face account is not authorized "
            "to access it."
        )
        print(
            "\nVisit the model page and request/obtain access:"
        )
        print(
            "https://huggingface.co/pyannote/"
            "speaker-diarization-community-1"
        )

    elif "HF_TOKEN" in error_message:
        print(
            "Hugging Face authentication token is missing."
        )

    else:
        print(
            "Could not load the pyannote diarization model."
        )
        print(f"\nError: {error_message}")

    print("\nDiarization will be skipped. Falling back to sequential transcription without speaker tags.")
    print("=" * 70)


# ============================================================
# 6. RUN DIARIZATION (Or Prepare Sequential Fallback Chunks)
# ============================================================

annotation = None

if diarization_available:
    print("\nRunning speaker diarization...")
    try:
        diarization = diarization_pipeline(AUDIO_PATH)
        annotation = diarization.speaker_diarization
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: DIARIZATION FAILED, falling back to sequential transcription.")
        print("=" * 70)
        print(str(e))
        print("=" * 70)


# ============================================================
# 7. RUN WHISPER ASR ON TURNS OR SEQUENTIAL CHUNKS
# ============================================================

print("\nRunning ASR...")

asr_segments = []

if annotation is not None:
    # Normal diarization flow
    for turn, _, speaker in tqdm(
        annotation.itertracks(yield_label=True),
        desc="Transcribing"
    ):
        turn_start = turn.start
        turn_end = turn.end
        start = turn_start

        while start < turn_end:
            end = min(start + MAX_CHUNK_SECONDS, turn_end)
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chunk = audio[start_sample:end_sample]

            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(input_features=input_features)

            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            if text:
                asr_segments.append({
                    "speaker": speaker,
                    "text": text
                })
            start = end
else:
    # Fallback sequential transcription flow
    total_duration = len(audio) / sr
    start = 0.0

    # Setup progress bar based on total chunks
    num_chunks = int(np.ceil(total_duration / MAX_CHUNK_SECONDS))
    pbar = tqdm(total=num_chunks, desc="Transcribing (No Diarization)")

    while start < total_duration:
        end = min(start + MAX_CHUNK_SECONDS, total_duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = audio[start_sample:end_sample]

        if len(chunk) > 0:
            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(input_features=input_features)

            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            if text:
                asr_segments.append({
                    "speaker": "SPEAKER_UNKNOWN",
                    "text": text
                })

        start = end
        pbar.update(1)
    pbar.close()


print(
    "\nTotal transcript segments:",
    len(asr_segments)
)


# ============================================================
# 8. SAVE RAW TRANSCRIPT
# ============================================================

print("\nSaving transcript...")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for segment in asr_segments:

        speaker = segment["speaker"]
        text = segment["text"]

        f.write(
            f"{speaker}: {text}\n"
        )


print("=" * 70)
print("TRANSCRIPTION COMPLETE")
print("=" * 70)

print(f"Saved transcript: {OUTPUT_PATH}")
print(
    f"Number of segments: {len(asr_segments)}"
)