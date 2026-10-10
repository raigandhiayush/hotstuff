# AI Meeting Assistant Pipeline

The **AI Meeting Assistant** converts meeting recordings into readable transcripts, meeting summaries, key decisions, and action items. It combines automatic speech recognition (ASR), speaker diarization, and large language models (LLMs) in a sequential pipeline, with a Gradio web interface for uploading audio and viewing the results.

## Architecture Overview

1. **Audio input:** Upload a `.wav` or `.mp3` meeting recording through the web interface.
2. **Stage 1 — Transcription and diarization:** Whisper with a custom LoRA adapter transcribes the audio, while Pyannote identifies speaker segments.
3. **Stage 2 — Transcript refinement:** Google Gemini cleans up the raw transcript while preserving its meaning and speaker labels.
4. **Stage 3 — Meeting information extraction:** Gemini generates an executive summary, key decisions, and action items from the refined transcript.
5. **Web interface:** `main.py` orchestrates the stages and displays the results in a Gradio dashboard.

## Setup and Running (Google Colab)

Google Colab with a GPU runtime is recommended for efficient audio processing.

### 1. Configure the Colab environment

1. Open a new Google Colab notebook.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU runtime).
3. Open the **Secrets** tab in the left sidebar and add these secrets. Enable **Notebook access** for both:
   - `HF_TOKEN` — your Hugging Face access token. Ensure you have accepted the required terms for the Pyannote model on Hugging Face.
   - `GEMINI_API_KEY` — your Google Gemini API key.

### 2. Upload and extract the project

Upload `hotstuff_colab.zip` to the Colab session, then run:

```bash
!unzip -q -o hotstuff_colab.zip -d hotstuff_project
```

### 3. Install dependencies

Install the packages listed in the project's requirements file:

```bash
!pip install -r hotstuff_project/requirements.txt
```

The commands above leave Colab's existing PyTorch installation in place unless a dependency explicitly changes it. If GPU-related package conflicts occur, check the installed PyTorch and CUDA versions before reinstalling packages.

### 4. Launch the pipeline

Load the secrets into environment variables and start the application:

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")

!python hotstuff_project/main.py
```

When the script starts successfully, it prints a Gradio URL ending in `gradio.live`. Open that link to access the dashboard.

> **Security note:** Keep API keys in Colab Secrets. Do not hard-code them into the source code or commit them to Git.

## Pipeline Stages

### Stage 1: Transcription and Speaker Diarization — `stage1.py`

**Purpose:** Convert meeting audio into text and identify the speakers associated with the audio segments.

- **Speaker diarization:** Uses `pyannote/speaker-diarization-community-1` to divide the recording into speaker-labelled segments, such as `SPEAKER_00` and `SPEAKER_01`.
- **Automatic speech recognition:** Uses the `whisper-small` model with the custom LoRA adapter in `whisper-small-lora-backup/` to transcribe the audio.
- **Output:** A raw transcript saved as `*_raw.txt`.

#### Whisper + LoRA fine-tuning details

The Whisper-small model was adapted for meeting-style English speech using the AMI Meeting Corpus.

| Setting | Value |
|---|---|
| Base model | Whisper-small |
| Training data | 10,000 AMI training samples and 1,000 validation samples |
| LoRA target modules | `q_proj` and `v_proj` |
| LoRA rank (`r`) | 8 |
| LoRA alpha (`α`) | 16 |
| Learning rate | `1e-4` |
| Batch size | 8 |
| Gradient accumulation | 2 steps |
| Training steps | 400 |

LoRA updates a comparatively small set of adapter parameters rather than fine-tuning all of Whisper's parameters. The aim is to improve transcription of meeting-style speech.

### Stage 2: Transcript Refinement — `meeting_refinement.py`

**Purpose:** Make the raw transcript easier to read while preserving what was said.

The raw transcript may contain filler words, repetitions, grammatical errors, or transcription mistakes. This stage sends the text to the Google Gemini API with instructions to:

- Correct grammar and punctuation.
- Remove unnecessary filler words and stutters where appropriate.
- Preserve the original meaning, context, and speaker tags.
- Avoid inventing information or changing the speakers' intent.

**Output:** A refined transcript saved as `*_refined.txt`.

### Stage 3: Meeting Minutes and Action Items — `stage3.py`

**Purpose:** Extract useful meeting information from the refined transcript.

The transcript is divided into chunks and sent to Gemini. The chunking process uses overlapping context to help preserve information that spans chunk boundaries. The intermediate outputs are then combined and sent through a final Gemini call to produce the consolidated result.

The output is intended to include:

- **Executive summary** — a concise overview of the meeting.
- **Key decisions** — decisions made during the discussion.
- **Action items** — tasks and, where identifiable, the people responsible for them.

**Outputs:**

- `*_record.md` — structured meeting record in Markdown.
- `*_record.json` — structured meeting record in JSON for possible downstream integration.

## Application Integration — `main.py`

`main.py` coordinates the pipeline and provides the Gradio interface. When a user uploads an audio file, it:

1. Saves the uploaded audio to the configured output or temporary location.
2. Calls `transcribe_stage1()` and waits for the raw transcript.
3. Reads the raw transcript and passes it to `refine_transcript()`.
4. Saves the refined transcript and passes it to `process_stage3()`.
5. Collects the raw transcript, refined transcript, and meeting record.
6. Displays the results in a three-column Gradio layout.

## Expected Outputs

For a processed recording, the pipeline produces the following files (exact names depend on the uploaded audio filename and the implementation):

| Output | Description |
|---|---|
| `*_raw.txt` | Transcript generated by Stage 1 |
| `*_refined.txt` | Cleaned transcript generated by Stage 2 |
| `*_record.md` | Meeting summary, decisions, and action items in Markdown |
| `*_record.json` | Structured meeting record in JSON |

## Troubleshooting

- **Missing Hugging Face credentials:** Confirm that `HF_TOKEN` exists in Colab Secrets and that Notebook access is enabled.
- **Pyannote access errors:** Confirm that you have accepted the model's required terms on Hugging Face and that your token has access.
- **Gemini authentication errors:** Check that `GEMINI_API_KEY` is correct and available in the notebook environment.
- **GPU not being used:** Confirm that a GPU runtime is selected under **Runtime → Change runtime type**. A GPU runtime does not guarantee that every pipeline component will use the GPU.
- **Dependency conflicts:** Review the installed package versions and the versions required by `requirements.txt` before changing Colab's preinstalled PyTorch/CUDA packages.
- **Gradio link not appearing:** Check the console for startup errors and ensure `main.py` reaches the Gradio launch step.

## Notes

- Keep API keys and tokens out of source control.
- The quality of the transcript and extracted action items depends on audio quality, diarization accuracy, and model outputs. Review the generated record before relying on it as an official meeting record.
