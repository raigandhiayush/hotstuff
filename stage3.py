import os
import json
import time
import argparse
from typing import Optional

from google import genai
from pydantic import BaseModel, Field

# ============================================================
# OUTPUT SCHEMAS
# ============================================================

class DiscussionPoint(BaseModel):
    topic: str = Field(description="The main topic discussed.")
    discussion: str = Field(description="Concise factual description of what was discussed.")

class Decision(BaseModel):
    decision: str = Field(description="A decision explicitly agreed upon during the meeting.")

class ActionItem(BaseModel):
    task: str = Field(description="An explicitly assigned, committed, or agreed task.")
    owner: Optional[str] = Field(default=None, description="Person explicitly assigned to the task. Null if not stated.")
    deadline: Optional[str] = Field(default=None, description="Explicitly stated deadline. Null if not stated.")

class MeetingRecord(BaseModel):
    summary: str = Field(description="Concise summary of the entire meeting.")
    minutes: list[DiscussionPoint] = Field(description="Organized meeting minutes.")
    decisions: list[Decision] = Field(description="Explicit decisions made during the meeting.")
    action_items: list[ActionItem] = Field(description="Explicit action items from the meeting.")

SYSTEM_PROMPT = """
You are Stage 3 of an AI meeting assistant.

Transform the refined English meeting transcript into a concise and accurate meeting record.

Use ONLY information explicitly stated in the transcript. Do not use outside knowledge or invent facts.

Preserve names, technical terminology, numbers, dates, project names, and other important terminology.

A proposal, suggestion, possibility, question, or discussion is NOT a decision.
Record a decision only when participants explicitly agree to it or clearly state that it has been decided.

Record an action item only when a task is explicitly assigned, explicitly committed to, or clearly agreed upon as something someone will do.

Never invent an action-item owner.
Never invent a deadline.
If the owner is not explicitly stated, use null.
If the deadline is not explicitly stated, use null.

The summary must describe what actually happened and must not introduce unsupported conclusions.

The minutes should contain the important topics discussed and must not invent or merge unrelated topics.

The final meeting record must faithfully represent the transcript.
"""

def process_stage3(input_file, final_json_file, final_md_file):
    MODEL_NAME = "gemini-3.5-flash-lite"

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable not found.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Transcript not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        transcript = f.read()

    final_prompt = f"{SYSTEM_PROMPT}\n\nMEETING TRANSCRIPT\n==================\n\n{transcript}\n\nEND MEETING TRANSCRIPT\n"

    schema = MeetingRecord.model_json_schema()
    
    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=final_prompt,
        generation_config={"thinking_level": "minimal"},
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": schema
        }
    )

    output_text = interaction.output_text
    if not output_text:
        raise RuntimeError("Gemini returned empty output.")

    meeting_record = MeetingRecord.model_validate_json(output_text)
    final_dict = meeting_record.model_dump()

    with open(final_json_file, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=2, ensure_ascii=False)

    md = ["# Meeting Record", "", "## Summary", "", final_dict["summary"], ""]
    md.extend(["## Minutes", ""])

    if final_dict["minutes"]:
        for i, point in enumerate(final_dict["minutes"], 1):
            md.extend([f"### {i}. {point['topic']}", "", point["discussion"], ""])
    else:
        md.extend(["No discussion points identified.", ""])

    md.extend(["## Decisions", ""])
    if final_dict["decisions"]:
        for i, decision in enumerate(final_dict["decisions"], 1):
            md.append(f"{i}. {decision['decision']}")
    else:
        md.append("No explicit decisions identified.")
    md.append("")

    md.extend(["## Action Items", ""])
    if final_dict["action_items"]:
        for i, action in enumerate(final_dict["action_items"], 1):
            owner = action["owner"] if action["owner"] else "Not specified"
            deadline = action["deadline"] if action["deadline"] else "Not specified"
            md.extend([f"### {i}. {action['task']}", "", f"- **Owner:** {owner}", f"- **Deadline:** {deadline}", ""])
    else:
        md.extend(["No explicit action items identified.", ""])

    markdown_output = "\n".join(md)
    with open(final_md_file, "w", encoding="utf-8") as f:
        f.write(markdown_output)

    return markdown_output, final_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--md", required=True)
    args = parser.parse_args()
    process_stage3(args.input, args.json, args.md)