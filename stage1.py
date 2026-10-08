import os
import sys
import argparse
import torch
import soundfile as sf
import numpy as np

from jiwer import wer
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from pyannote.audio import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def transcribe_stage1(audio_path, output_path, backup_dir=None):
    if backup_dir is None:
        backup_dir = os.path.join(BASE_DIR, "whisper-small-lora-backup")
    MODEL_NAME = "openai/whisper-small"
    
    if not os.path.isdir(backup_dir):
        raise FileNotFoundError(f"LoRA model directory not found: {os.path.abspath(backup_dir)}")

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {os.path.abspath(audio_path)}")

    MAX_CHUNK_SECONDS = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Meeting Transcription Pipeline")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Audio:  {audio_path}")

    print("\nLoading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    print("Loading base Whisper model...")
    # Use fp16 for much faster generation on T4 GPUs
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype).to(device)
    base_model.generation_config.language = "english"
    base_model.generation_config.task = "transcribe"

    if not hasattr(base_model, "_prepare_encoder_decoder_kwargs_for_generation"):
        base_model._prepare_encoder_decoder_kwargs_for_generation = lambda *args, **kwargs: {}

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, backup_dir)
    model.eval()

    print("\nLoading audio...")
    import librosa
    audio, original_sr = librosa.load(audio_path, sr=None, mono=False)
    
    if audio.ndim > 1:
        audio = audio.mean(axis=0)  # librosa stereo is (channels, samples)
    audio = audio.astype(np.float32)

    TARGET_SR = 16000
    if original_sr != TARGET_SR:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, TARGET_SR, original_sr).astype(np.float32)
        sr = TARGET_SR
    else:
        sr = original_sr

    print("\nLoading pyannote diarization model...")
    HF_TOKEN = os.environ.get("HF_TOKEN")
    diarization_pipeline = None
    diarization_available = False

    try:
        if HF_TOKEN is None:
            raise RuntimeError("HF_TOKEN environment variable is not set.")
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)
        if diarization_pipeline:
            diarization_pipeline.to(torch.device(device))
            diarization_available = True
    except Exception as e:
        print("WARNING: SPEAKER DIARIZATION UNAVAILABLE", str(e))

    annotation = None
    if diarization_available:
        try:
            diarization = diarization_pipeline(audio_path)
            annotation = diarization.speaker_diarization
        except Exception as e:
            print("ERROR: DIARIZATION FAILED, falling back to sequential transcription.", str(e))

    print("\nRunning ASR...")
    asr_segments = []

    if annotation is not None:
        for turn, _, speaker in tqdm(annotation.itertracks(yield_label=True), desc="Transcribing"):
            start = turn.start
            while start < turn.end:
                end = min(start + MAX_CHUNK_SECONDS, turn.end)
                start_sample, end_sample = int(start * sr), int(end * sr)
                chunk = audio[start_sample:end_sample]
                if len(chunk) > 0:
                    inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
                    with torch.no_grad():
                        predicted_ids = model.generate(input_features=inputs.input_features.to(device, dtype=torch_dtype))
                    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                    if text:
                        asr_segments.append({"speaker": speaker, "text": text})
                start = end
    else:
        total_duration = len(audio) / sr
        start = 0.0
        while start < total_duration:
            end = min(start + MAX_CHUNK_SECONDS, total_duration)
            start_sample, end_sample = int(start * sr), int(end * sr)
            chunk = audio[start_sample:end_sample]
            if len(chunk) > 0:
                inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
                with torch.no_grad():
                    predicted_ids = model.generate(input_features=inputs.input_features.to(device, dtype=torch_dtype))
                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                if text:
                    asr_segments.append({"speaker": "SPEAKER_UNKNOWN", "text": text})
            start = end

    print("\nSaving transcript...")
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in asr_segments:
            f.write(f"{seg['speaker']}: {seg['text']}\n")
    print(f"Saved transcript: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    transcribe_stage1(args.audio, args.output)
