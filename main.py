import os
import gradio as gr
import shutil
from pathlib import Path
from stage1 import transcribe_stage1
from meeting_refinement import refine_transcript
from stage3 import process_stage3

def process_meeting(audio_file):
    if audio_file is None:
        return "No file uploaded.", "", "", None, None, None

    if isinstance(audio_file, str):
        audio_path = audio_file
    else:
        audio_path = audio_file.name
    base_name = Path(audio_path).stem
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    
    raw_transcript_file = str(out_dir / f"{base_name}_raw.txt")
    refined_transcript_file = str(out_dir / f"{base_name}_refined.txt")
    json_file = str(out_dir / f"{base_name}_record.json")
    md_file = str(out_dir / f"{base_name}_record.md")

    try:
        # Stage 1: Transcription
        transcribe_stage1(audio_path, raw_transcript_file)
        with open(raw_transcript_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Stage 2: Refinement
        # Load keys from Colab userdata if available
        try:
            from google.colab import userdata
            if not os.environ.get("GEMINI_API_KEY"):
                os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
            if not os.environ.get("HF_TOKEN"):
                os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        except Exception as e:
            print(f"Warning: Could not fetch from Colab userdata ({e}). Ensure env vars are set.")
            pass

        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is not set in environment variables or Colab secrets.")
        
        # We pass the raw_text to refinement
        refine_result = refine_transcript(raw_text=raw_text, run_dir=out_dir)
        refined_text = refine_result.get("refined_transcript", "")
        
        with open(refined_transcript_file, "w", encoding="utf-8") as f:
            f.write(refined_text)

        # Stage 3: Meeting Minutes
        markdown_output, _ = process_stage3(refined_transcript_file, json_file, md_file)

        return raw_text, refined_text, markdown_output, raw_transcript_file, refined_transcript_file, md_file
    
    except Exception as e:
        return f"Error occurred: {str(e)}", "", "", None, None, None

def main():
    with gr.Blocks(title="AI Meeting Assistant Dashboard") as demo:
        gr.Markdown("# AI Meeting Assistant Dashboard")
        gr.Markdown("Upload an audio file (MP3 or WAV) to process the meeting and get transcriptions, refinement, and minutes.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.File(label="Upload Audio File", file_types=[".mp3", ".wav"])
                process_btn = gr.Button("Process Meeting", variant="primary")
            
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Raw Transcript")
                raw_out = gr.TextArea(label="Raw Transcript", interactive=False)
                raw_download = gr.File(label="Download Raw Transcript")
            with gr.Column():
                gr.Markdown("### Refined Transcript")
                refined_out = gr.TextArea(label="Refined Transcript", interactive=False)
                refined_download = gr.File(label="Download Refined Transcript")
            with gr.Column():
                gr.Markdown("### Meeting Minutes & Action Items")
                md_out = gr.Markdown()
                md_download = gr.File(label="Download Meeting Record (Markdown)")

        process_btn.click(
            fn=process_meeting,
            inputs=[audio_input],
            outputs=[raw_out, refined_out, md_out, raw_download, refined_download, md_download]
        )
    
    demo.launch(share=True) # Share=True creates a public gradio link which is good for Colab

if __name__ == "__main__":
    main()
