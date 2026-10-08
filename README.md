# AI Meeting Assistant Pipeline

Welcome to the AI Meeting Assistant! This project provides an end-to-end pipeline that takes an audio recording of a meeting and converts it into a clean, refined transcript along with structured meeting minutes and action items.

## Brief Overall Architecture

The pipeline is designed as a multi-stage sequential architecture that leverages state-of-the-art AI models:

1. **Audio Input**: An audio file (`.wav`, `.mp3`) is uploaded via the web interface.
2. **Stage 1 (Audio processing)**: The audio is passed through an Automatic Speech Recognition (ASR) model (Whisper + LoRA) and a speaker diarization model (Pyannote).
3. **Stage 2 (Text Refinement)**: The raw, slightly messy text is passed to an LLM (Google Gemini) to clean up stutters and fix grammatical mistakes.
4. **Stage 3 (Information Extraction)**: The cleaned text is passed to the LLM again to extract meeting minutes, summaries, and action items.
5. **Web UI**: Everything is orchestrated by a Gradio dashboard (`main.py`) which displays the results back to the user.

\---

## Complete Setup \& Running Steps (Google Colab)

To run this project efficiently, it is highly recommended to use Google Colab with a GPU.

### 1\. Configure the Colab Environment

* Open a new Google Colab notebook.
* Go to **Runtime > Change runtime type** and select **T4 GPU**.
* Open the **Secrets** tab on the left sidebar. Add two secrets and grant them "Notebook access":

  * `HF\_TOKEN`: Your Hugging Face API token (Make sure you have accepted the Pyannote terms on HuggingFace).
  * `GEMINI\_API\_KEY`: Your Google Gemini API key.

### 2\. Upload and Extract Files

Upload this zip file to your Colab workspace and extract it:

```bash
!unzip -q -o hotstuff\_colab.zip -d hotstuff\_project
```

### 3\. Install Dependencies

Install the required packages. (Note: Colab's default PyTorch is used to ensure GPU compatibility).

```bash
!pip install -r hotstuff\_project/requirements.txt
```

### 4\. Run the Pipeline

Execute the main script. The script will automatically pull your secrets and launch a public web link.

```python
import os
from google.colab import userdata

os.environ\["HF\_TOKEN"] = userdata.get('HF\_TOKEN')
os.environ\["GEMINI\_API\_KEY"] = userdata.get('GEMINI\_API\_KEY')

!python hotstuff\_project/main.py
```

Click the `gradio.live` link printed in the console to open the dashboard!

\---

## Pipeline Stages in Detail

### Stage 1: Transcription and Diarization (`stage1.py`)

**Goal:** Convert spoken audio into raw text and identify who is speaking.

* **Speaker Diarization:** Uses `pyannote/speaker-diarization-community-1` to segment the audio by speaker (e.g., SPEAKER\_00, SPEAKER\_01).
* **Automatic Speech Recognition (ASR):** Uses OpenAI's `whisper-small` model, enhanced with a custom fine-tuned LoRA adapter (located in `whisper-small-lora-backup/`) to accurately transcribe the audio segments into text.
* Fine-Tuning Method — Whisper + LoRA
* Base model: Pre-trained Whisper-small, fine-tuned on the AMI Meeting Corpus for meeting-specific English speech recognition.
* Parameter-efficient fine-tuning: Applied LoRA to the Whisper query (q\_proj) and value (v\_proj) projection layers, using rank r=8 and α=16.
* Training setup: Used 10,000 AMI training samples + 1,000 validation samples, with learning rate 1e-4, batch size 8, gradient accumulation 2, and 400 training steps.
* Objective: Optimize transcription performance on meeting-style speech while training only a small number of additional LoRA parameters instead of updating the entire Whisper model.
* **Output:** A raw transcript file (`\*\_raw.txt`).

### Stage 2: Transcript Refinement (`meeting\_refinement.py`)

**Goal:** Clean up the raw transcript to make it readable.

* **Process:** The raw text often contains "ums", "ahs", stuttering, or minor transcription errors. This stage passes the raw text to the Google Gemini API.
* **Prompt Engineering:** Gemini is instructed to fix grammar and remove filler words while strictly preserving the original intent, context, and speaker tags.
* **Output:** A beautifully formatted, readable transcript (`\*\_refined.txt`).

### Stage 3: Meeting Minutes \& Action Items (`stage3.py`)

**Goal:** Extract actionable intelligence from the meeting.

* **Process:** The *refined* transcript is sent to the Gemini API in chunks.
* Several chunks are sent to Gemini API with overlap with previous window to preserve context.
* Then finally outputs of all such calls is again fed to Gemini API to furnish final ouputs.
* **Output Generation:** The LLM generates a structured summary consisting of:

  * Executive Summary
  * Key Decisions
  * Action Items (who needs to do what)
* **Output:** A structured Markdown file (`\*\_record.md`) and a JSON record (`\*\_record.json`) for potential database integration.

\---

## Overall Integration Process (`main.py`)

The `main.py` script acts as the orchestrator for the entire pipeline using **Gradio**.

When a user uploads a file to the web UI:

1. `main.py` saves the audio to a temporary output folder.
2. It triggers `transcribe\_stage1()`. It waits for the raw `.txt` file to be generated.
3. It immediately reads that raw text and passes it to `refine\_transcript()`.
4. It takes the returned refined text, saves it, and passes it to `process\_stage3()`.
5. Once all stages complete, `main.py` gathers the raw text, refined text, and markdown summary, and simultaneously updates the Gradio Web UI to display everything to the user in a clean, three-column layout.

