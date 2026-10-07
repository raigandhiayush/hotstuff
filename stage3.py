# ============================================================
# STAGE 3 — AI MEETING ASSISTANT
# Gemini 3.5 Flash-Lite
# ============================================================

!pip install -q -U google-genai pydantic

import os
import json
import time
from typing import Optional

from google import genai
from pydantic import BaseModel, Field

print("=" * 70)
print("STAGE 3 — AI MEETING ASSISTANT")
print("=" * 70)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"
INPUT_FILE = "/content/refined_transcript.txt"
FINAL_JSON_FILE = "/content/meeting_record.json"
FINAL_MD_FILE = "/content/meeting_record.md"

print("\n" + "=" * 70)
print("CONFIGURATION")
print("=" * 70)
print(f"[INFO] Model           : {MODEL_NAME}")
print("[INFO] Thinking        : minimal")
print(f"[INFO] Input           : {INPUT_FILE}")
print(f"[INFO] Final JSON      : {FINAL_JSON_FILE}")
print(f"[INFO] Final Markdown  : {FINAL_MD_FILE}")


# ============================================================
# 2. GEMINI API KEY
# ============================================================

print("\n" + "=" * 70)
print("CHECKING GEMINI API KEY")
print("=" * 70)

try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Add it to Google Colab Secrets "
        "with the name GEMINI_API_KEY and enable notebook access."
    )

print("[OK] Gemini API key found.")


# ============================================================
# 3. INITIALIZE GEMINI
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING GEMINI")
print("=" * 70)

client = genai.Client(api_key=GEMINI_API_KEY)

print("[OK] Gemini client initialized.")


# ============================================================
# 4. LOAD TRANSCRIPT
# ============================================================

print("\n" + "=" * 70)
print("LOADING REFINED TRANSCRIPT")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Transcript not found: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    transcript = f.read()

print("[OK] Transcript loaded.")
print(f"[INFO] Characters : {len(transcript):,}")
print(f"[INFO] Words      : {len(transcript.split()):,}")


# ============================================================
# 5. OUTPUT SCHEMAS
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


print("[OK] Output schema created.")


# ============================================================
# 6. STAGE 3 PROMPT
# ============================================================

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


# ============================================================
# 7. CREATE REQUEST
# ============================================================

final_prompt = f"""
{SYSTEM_PROMPT}

MEETING TRANSCRIPT
==================

{transcript}

END MEETING TRANSCRIPT
"""


# ============================================================
# 8. GEMINI REQUEST
# ============================================================

print("\n" + "=" * 70)
print("SENDING TRANSCRIPT TO GEMINI")
print("=" * 70)

schema = MeetingRecord.model_json_schema()
start_time = time.time()

try:
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
except Exception as e:
    print(f"[ERROR] Gemini request failed: {type(e).__name__}: {e}")
    raise

elapsed = time.time() - start_time

print("[OK] Gemini response received.")
print(f"[INFO] Inference time: {elapsed:.2f} seconds")


# ============================================================
# 9. VALIDATE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING GEMINI OUTPUT")
print("=" * 70)

output_text = interaction.output_text

if not output_text:
    raise RuntimeError("Gemini returned empty output.")

print(f"[INFO] Output characters: {len(output_text):,}")

try:
    meeting_record = MeetingRecord.model_validate_json(output_text)
except Exception as e:
    print("[ERROR] Output validation failed.")
    print(output_text)
    raise e

print("[OK] JSON structure validated.")
print("[OK] Pydantic validation successful.")
print(f"[INFO] Minutes      : {len(meeting_record.minutes)}")
print(f"[INFO] Decisions    : {len(meeting_record.decisions)}")
print(f"[INFO] Action items : {len(meeting_record.action_items)}")


# ============================================================
# 10. SAVE JSON
# ============================================================

print("\n" + "=" * 70)
print("SAVING JSON")
print("=" * 70)

final_dict = meeting_record.model_dump()

with open(FINAL_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(final_dict, f, indent=2, ensure_ascii=False)

print(f"[OK] Saved: {FINAL_JSON_FILE}")


# ============================================================
# 11. CREATE MARKDOWN
# ============================================================

print("\n" + "=" * 70)
print("CREATING MARKDOWN")
print("=" * 70)

md = ["# Meeting Record", "", "## Summary", "", final_dict["summary"], ""]

md.extend(["## Minutes", ""])

if final_dict["minutes"]:
    for i, point in enumerate(final_dict["minutes"], 1):
        md.extend([
            f"### {i}. {point['topic']}",
            "",
            point["discussion"],
            ""
        ])
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

        md.extend([
            f"### {i}. {action['task']}",
            "",
            f"- **Owner:** {owner}",
            f"- **Deadline:** {deadline}",
            ""
        ])
else:
    md.extend(["No explicit action items identified.", ""])

markdown_output = "\n".join(md)

with open(FINAL_MD_FILE, "w", encoding="utf-8") as f:
    f.write(markdown_output)

print(f"[OK] Saved: {FINAL_MD_FILE}")


# ============================================================
# 12. DISPLAY FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL MEETING RECORD")
print("=" * 70)
print()
print(markdown_output)


# ============================================================
# 13. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STAGE 3 COMPLETE")
print("=" * 70)
print("[OK] Meeting record generated successfully.")
print(f"[OK] JSON     : {FINAL_JSON_FILE}")
print(f"[OK] Markdown : {FINAL_MD_FILE}")