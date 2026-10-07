"""
meeting_refinement.py
=====================
Stage 1 (ASR) + Stage 2 (domain-aware transcript refinement) of the meeting
pipeline, converted from the Kaggle notebook into an importable module.

The UI (Streamlit / Gradio / Flask ...) only needs:

    from meeting_refinement import process_audio, refine_transcript, PipelineError

    result = process_audio("/tmp/upload.mp3", api_key=key, on_progress=cb)
    result["raw_transcript"], result["refined_transcript"], result["diff_html"] ...

Config (all optional, via environment variables or a .env file):
    GEMINI_API_KEY      required for refinement
    REFINE_MODEL_ID     default "gemini-3.1-flash-lite"
    WHISPER_MODEL       default "small.en"
    MEETING_DATA_DIR    default <this file's folder>/outputs
    HF_HOME             default <this file's folder>/.cache/huggingface
"""
from __future__ import annotations

import argparse
import difflib
import gc
import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

# ----------------------------------------------------------------------------
# PATHS  (no /kaggle anywhere; everything is relative to this file or env vars)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

try:  # optional: load GEMINI_API_KEY etc. from a local .env file
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

DATA_DIR = Path(os.environ.get("MEETING_DATA_DIR", BASE_DIR / "outputs")).resolve()
CACHE_DIR = DATA_DIR / "cache"
# Must be set BEFORE faster_whisper / huggingface_hub is imported.
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".cache" / "huggingface"))

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
MODEL_ID = os.environ.get("REFINE_MODEL_ID", "gemini-3.1-flash-lite")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small.en")

CHUNK_SIZE = 8
OVERLAP = 0
MAX_WORKERS = int(os.environ.get("REFINE_WORKERS", "4"))  # parallel Gemini calls
API_RETRIES = 5
MIN_CONFIDENCE = 0.80
EXTRA_GLOSSARY: list = []
CACHE_VERSION = "speaker-format-v3-gemini-flash"

SUPPORTED_AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".webm",
}

ProgressCB = Callable[[str, float, str], None]  # (stage, fraction 0..1, message)


class PipelineError(Exception):
    """Error whose message is safe to show directly to the user in the UI."""


def _noop(stage: str, frac: float, msg: str) -> None:
    pass


# ----------------------------------------------------------------------------
# API KEY
# ----------------------------------------------------------------------------
def resolve_api_key(api_key: Optional[str] = None) -> str:
    """Order: explicit argument (e.g. from st.secrets / UI field) > env var / .env."""
    key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise PipelineError(
            "GEMINI_API_KEY is not set. Put it in a .env file, export it as an "
            "environment variable, or enter it in the app."
        )
    return key


# ----------------------------------------------------------------------------
# STAGE 1: SPEECH TO TEXT (faster-whisper)
# ----------------------------------------------------------------------------
_ASR_MODEL = None  # loaded once per process (the notebook reloaded it on every call)


def _device() -> str:
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def _get_asr():
    global _ASR_MODEL
    if _ASR_MODEL is None:
        from faster_whisper import WhisperModel
        dev = _device()
        _ASR_MODEL = WhisperModel(
            WHISPER_MODEL, device=dev,
            compute_type="float16" if dev == "cuda" else "int8",
        )
    return _ASR_MODEL


def validate_audio_file(path) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise PipelineError(f"Audio file not found: {p.name}")
    if p.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
        raise PipelineError(
            f"Unsupported file type '{p.suffix}'. Supported: "
            + ", ".join(sorted(SUPPORTED_AUDIO_EXTS))
        )
    if p.stat().st_size == 0:
        raise PipelineError("The uploaded file is empty.")
    return p


def transcribe_audio(path, on_progress: ProgressCB = _noop) -> str:
    """Returns the raw transcript, one Whisper segment per line."""
    p = validate_audio_file(path)
    on_progress("transcribe", 0.0, "Loading speech-to-text model...")
    asr = _get_asr()
    on_progress("transcribe", 0.1, "Transcribing audio...")
    try:
        segs, info = asr.transcribe(str(p), language="en", beam_size=1, vad_filter=True)
        total = max(getattr(info, "duration", 0) or 0, 1e-6)
        lines = []
        for s in segs:  # generator: decoding errors surface here
            t = s.text.strip()
            if t:
                lines.append(t)
            on_progress("transcribe", min(0.1 + 0.9 * s.end / total, 1.0), "Transcribing audio...")
    except Exception as e:
        raise PipelineError(f"Could not read this audio file (corrupt or unsupported codec): {e}")
    finally:
        gc.collect()
    if not lines:
        raise PipelineError("No speech detected in the audio file.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# SEGMENTS
# ----------------------------------------------------------------------------
SPEAKER_RE = re.compile(r"^\s*(SPEAKER_\d+)\s*:\s*(.*)$", re.IGNORECASE)


def load_segments(text: str):
    """One segment per non-empty line. 'SPEAKER_n:' prefix is optional (Whisper
    output has none; the old notebook silently dropped such lines)."""
    segments = []
    for i, line in enumerate((l.strip() for l in text.splitlines()), start=1):
        if not line:
            continue
        m = SPEAKER_RE.match(line)
        speaker, content = (m.group(1).upper(), m.group(2).strip()) if m else ("", line)
        if content:
            segments.append({"id": f"S{i:03d}", "speaker": speaker, "text": content})
    return segments


def make_chunks(segments, chunk_size=12, overlap=0):
    chunks, start = [], 0
    while start < len(segments):
        end = min(start + chunk_size, len(segments))
        chunks.append(segments[start:end])
        if end == len(segments):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_to_text(chunk):
    return "\n".join(
        f'{s["id"]} | {s["speaker"]} | {s["text"]}' if s["speaker"] else f'{s["id"]} | {s["text"]}'
        for s in chunk
    )


def segments_to_text(segs):
    return "\n".join(f'{s["speaker"]}: {s["text"]}' if s["speaker"] else s["text"] for s in segs)


# ----------------------------------------------------------------------------
# GEMINI HELPERS
# ----------------------------------------------------------------------------
def extract_json(text):
    if text is None:
        raise ValueError("Empty model response")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        starts = [p for p in (text.find("["), text.find("{")) if p != -1]
        if not starts:
            raise
        start = min(starts)
        end = max(text.rfind("]"), text.rfind("}"))
        if end <= start:
            raise
        return json.loads(text[start:end + 1])


class GeminiLLM:
    def __init__(self, api_key: str, model_id: str = MODEL_ID):
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id

    def call(self, system_prompt, user_prompt, max_output_tokens=1024) -> str:
        types, last = self._types, None
        for attempt in range(API_RETRIES):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=max_output_tokens,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                    ),
                )
                if not resp.text or not resp.text.strip():
                    raise ValueError("Gemini returned an empty response")
                return resp.text
            except Exception as e:
                last = e
                if attempt < API_RETRIES - 1:
                    time.sleep(2 ** attempt)
        raise PipelineError(f"Gemini request failed after {API_RETRIES} attempts: {last}")

    def batch(self, system_prompt, prompts, max_new_tokens=1024):
        with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as ex:
            return list(ex.map(lambda p: self.call(system_prompt, p, max_new_tokens), prompts))


# ----------------------------------------------------------------------------
# LEXICON
# ----------------------------------------------------------------------------
LEXICON_SYSTEM = """
You are a terminology extraction system.

Extract domain-specific terminology from the transcript: software/framework names, libraries, tools,
platforms, technical concepts, acronyms, project names, organization names, people's names, product
names, version names. Skip ordinary everyday words.

Do NOT invent terms.

Return ONLY valid JSON.
Format:

[
  {"term": "PyTorch", "type": "framework"}
]
"""


def sample_blocks(segs, n_blocks=4, block_size=40):
    n = len(segs)
    if n <= block_size:
        return [segs]
    starts = sorted({int(i * (n - block_size) / max(n_blocks - 1, 1)) for i in range(n_blocks)})
    return [segs[s:s + block_size] for s in starts]


def build_lexicon(segments, llm: GeminiLLM):
    prompts = [
        "Extract the technical/domain terminology from this meeting excerpt.\n\n"
        f"TRANSCRIPT:\n{chunk_to_text(b)}"
        for b in sample_blocks(segments)
    ]
    responses = llm.batch(LEXICON_SYSTEM, prompts, 1024)

    cands = [{"term": t, "type": "glossary"} for t in EXTRA_GLOSSARY]
    for r in responses:
        try:
            cands += extract_json(r)
        except Exception:
            pass  # a bad block just contributes nothing

    lexicon = {}
    for item in cands:
        if not isinstance(item, dict) or not str(item.get("term", "")).strip():
            continue
        term = item["term"].strip()
        if any(c.isupper() or c.isdigit() for c in term) or " " in term:
            lexicon.setdefault(term.lower(), {"term": term, "type": item.get("type", "")})
    return lexicon


# ----------------------------------------------------------------------------
# MATCHING / VALIDATION  (logic unchanged from the notebook)
# ----------------------------------------------------------------------------
from rapidfuzz import fuzz  # noqa: E402
import jellyfish  # noqa: E402


def lexical_similarity(a, b):
    return fuzz.ratio(a.lower(), b.lower())


def phonetic_repr(text):
    return " ".join(jellyfish.metaphone(w) for w in re.findall(r"[A-Za-z]+", text.lower()))


def phonetic_similarity(a, b):
    pa, pb = phonetic_repr(a), phonetic_repr(b)
    if not pa or not pb:
        return 0.0
    return fuzz.ratio(pa, pb)


def extract_numbers(text):
    return re.findall(r"\d+(?:\.\d+)?", text)


PROTECTED_WORDS = {
    "not", "no", "nor", "never", "none", "nothing", "without", "neither",
    "cannot", "can't", "won't", "wouldn't", "couldn't", "shouldn't", "mustn't",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't",
    "might", "may", "could", "would", "should", "must", "shall", "will", "can",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "percent",
}


def protected_tokens(text):
    text = text.replace("\u2019", "'")
    return [w for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower()) if w in PROTECTED_WORDS]


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")


def normalise_token(token):
    token = token.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    return token.strip(".,!?;:\"'()[]{}")


def tokenise_with_spans(text):
    return [
        {"token": m.group(0), "norm": normalise_token(m.group(0)),
         "start": m.start(), "end": m.end()}
        for m in TOKEN_RE.finditer(text)
    ]


def normalise_phrase(text):
    return " ".join(t["norm"] for t in tokenise_with_spans(text))


def find_edit_match(segment_text, original):
    if not original or not original.strip():
        return None
    source_tokens = tokenise_with_spans(segment_text)
    original_tokens = tokenise_with_spans(original)
    if not source_tokens or not original_tokens:
        return None

    original_norm = normalise_phrase(original)
    original_words = original_norm.split()
    n = len(original_words)

    # LEVEL 1: exact token-normalised match
    exact = []
    for i in range(len(source_tokens) - n + 1):
        if [source_tokens[j]["norm"] for j in range(i, i + n)] == original_words:
            exact.append((source_tokens[i]["start"], source_tokens[i + n - 1]["end"]))
    if len(exact) == 1:
        s, e = exact[0]
        return {"start": s, "end": e, "matched_text": segment_text[s:e],
                "match_type": "normalised-exact", "score": 100.0}
    if len(exact) > 1:
        return None

    # LEVEL 2: conservative fuzzy contiguous match
    candidates = []
    min_len = max(1, n - 2)
    max_len = min(len(source_tokens), n + 2)
    for wl in range(min_len, max_len + 1):
        for i in range(len(source_tokens) - wl + 1):
            j = i + wl
            cand_text = segment_text[source_tokens[i]["start"]:source_tokens[j - 1]["end"]]
            cand_norm = normalise_phrase(cand_text)
            lexical = fuzz.ratio(original_norm, cand_norm)
            phonetic = phonetic_similarity(original, cand_text)
            combined = 0.70 * lexical + 0.30 * phonetic
            cw = len(cand_norm.split())
            if max(len(original_words), cw) == 0:
                continue
            combined *= min(len(original_words), cw) / max(len(original_words), cw)
            candidates.append({
                "start": source_tokens[i]["start"], "end": source_tokens[j - 1]["end"],
                "matched_text": cand_text, "lexical": lexical,
                "phonetic": phonetic, "score": combined,
            })
    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    distinct = []
    for c in candidates:
        if not any(not (c["end"] <= d["start"] or c["start"] >= d["end"]) for d in distinct):
            distinct.append(c)
    distinct.sort(key=lambda x: x["score"], reverse=True)

    best = distinct[0]
    second = distinct[1]["score"] if len(distinct) > 1 else 0.0
    if n == 1:
        if best["score"] < 90:
            return None
    else:
        if best["score"] < 78:
            return None
        if len(distinct) > 1 and best["score"] - second < 8:
            return None

    return {"start": best["start"], "end": best["end"], "matched_text": best["matched_text"],
            "match_type": "fuzzy", "score": best["score"],
            "lexical": best["lexical"], "phonetic": best["phonetic"]}


def validate_edit(edit, segment_text, lexicon_terms, min_confidence=MIN_CONFIDENCE):
    try:
        original = edit["original"]
        corrected = edit["corrected"]
        confidence = float(edit.get("confidence", 0))
        assert isinstance(original, str) and isinstance(corrected, str)
    except Exception:
        return False, "malformed edit"

    if not original.strip() or not corrected.strip():
        return False, "empty edit"
    if original.strip().lower() == corrected.strip().lower():
        return False, "no-op edit"

    match = find_edit_match(segment_text, original)
    if match is None:
        return False, "original not found uniquely"
    if confidence < min_confidence:
        return False, "low confidence"
    if extract_numbers(original) != extract_numbers(corrected):
        return False, "number changed"
    if protected_tokens(original) != protected_tokens(corrected):
        return False, "negation/modality changed"

    lexical = lexical_similarity(original, corrected)
    phonetic = phonetic_similarity(original, corrected)
    known = corrected.lower() in lexicon_terms
    if not (lexical >= 55 or (known and phonetic >= 40)):
        return False, "edit too dissimilar"

    return True, {"match": match, "lexical": lexical, "phonetic": phonetic}


def run_self_test() -> bool:
    segs = {
        "T1": "Okay so we need to deploy the pie torch model on the server.",
        "T2": "We won't deploy the new image until testing is complete.",
        "T3": "Accuracy is 92 percent on the test set.",
        "T4": "The model is slow.",
    }
    lex = {"pytorch", "kubernetes"}
    tests = [("T1", "pie torch", "PyTorch", True), ("T2", "won't deploy", "will deploy", False),
             ("T3", "92 percent", "95 percent", False), ("T4", "model", "Kubernetes", False)]
    all_ok = True
    for sid, o, c, expected in tests:
        ok, info = validate_edit({"original": o, "corrected": c, "confidence": 0.99}, segs[sid], lex)
        all_ok &= (ok == expected)
        print("PASS" if ok == expected else "FAIL", "|", o, "->", c, "| accepted:", ok)
    return all_ok


# ----------------------------------------------------------------------------
# STAGE 2: REFINEMENT
# ----------------------------------------------------------------------------
REFINEMENT_SYSTEM = """
You are a highly conservative transcript refinement engine.

Your job is NOT to rewrite or paraphrase the transcript.

Identify only high-confidence transcription errors.

Allowed corrections:
1. Technical terminology
2. Acronyms
3. Proper nouns
4. Framework/tool/project names
5. Obvious punctuation restoration
6. Minor speech artifacts when they hurt readability

Rules:

- Make the smallest possible edit.
- Preserve meaning exactly.
- Never invent information.
- Never alter numbers, dates, percentages or versions.
- NEVER alter negation.
- NEVER alter modality.
- Do not change "will" to "would", "can" to "could", etc.
- If uncertain, DO NOTHING.
- The corrected text must be acoustically or orthographically plausible.
- The "original" field must be copied EXACTLY from the transcript.
- Only return actual corrections.
- If there are no corrections, return [].

Return ONLY valid JSON:

[
  {
    "segment_id": "S001",
    "original": "pie torch",
    "corrected": "PyTorch",
    "reason": "framework name",
    "confidence": 0.98
  }
]
"""


def _build_prompt(chunk, lexicon_text):
    return f"""
GLOBAL DOMAIN LEXICON:
{lexicon_text}

MEETING EXCERPT:
{chunk_to_text(chunk)}

Find only high-confidence transcription errors.

Return JSON edits only.
"""


def refine_transcript(
    raw_text: str,
    api_key: Optional[str] = None,
    run_dir: Optional[Path] = None,
    on_progress: ProgressCB = _noop,
    use_cache: bool = True,
    model_id: str = MODEL_ID,
) -> dict:
    """Stage 2. Takes raw text, returns raw/refined transcripts, edits, diff, etc."""
    if not raw_text or not raw_text.strip():
        raise PipelineError("The transcript is empty.")

    segments = load_segments(raw_text)
    if not segments:
        raise PipelineError("No readable text found in the transcript.")
    chunks = make_chunks(segments, CHUNK_SIZE, OVERLAP)

    llm = GeminiLLM(resolve_api_key(api_key), model_id)

    on_progress("refine", 0.0, "Extracting domain terminology...")
    lexicon = build_lexicon(segments, llm)
    lexicon_text = "\n".join(f'- {v["term"]} ({v["type"]})' for v in list(lexicon.values())[:60])

    # ---- proposals (cached per input, so re-running the same transcript is free)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "edits_cache.json"
    key = hashlib.md5((CACHE_VERSION + model_id + str(CHUNK_SIZE) + str(OVERLAP)
                       + raw_text[:200000] + REFINEMENT_SYSTEM + lexicon_text).encode()).hexdigest()
    cache = {"key": key, "chunks": {}}
    if use_cache and cache_path.exists():
        try:
            saved = json.loads(cache_path.read_text())
            if saved.get("key") == key:
                cache = saved
        except Exception:
            pass

    todo = [i for i in range(len(chunks)) if str(i) not in cache["chunks"]]
    group_size = MAX_WORKERS * 2
    for g in range(0, len(todo), group_size):
        group = todo[g:g + group_size]
        on_progress("refine", 0.1 + 0.8 * g / max(len(todo), 1),
                    f"Refining transcript ({g}/{len(todo)} chunks)...")
        responses = llm.batch(REFINEMENT_SYSTEM,
                              [_build_prompt(chunks[i], lexicon_text) for i in group], 1024)
        for i, r in zip(group, responses):
            try:
                cache["chunks"][str(i)] = extract_json(r)
            except Exception:
                pass  # bad JSON for a chunk -> no edits for it
        cache_path.write_text(json.dumps(cache))

    all_edits = [e for i in range(len(chunks)) for e in cache["chunks"].get(str(i), [])
                 if isinstance(e, dict)]

    # ---- validate
    seg_lookup = {s["id"]: s["text"] for s in segments}
    lexicon_terms = set(lexicon.keys())
    unique = {}
    for e in all_edits:
        try:
            unique[(e["segment_id"], e["original"].lower(), e["corrected"].lower())] = e
        except Exception:
            pass

    valid_edits, rejected_edits = [], []
    for e in unique.values():
        if e["segment_id"] not in seg_lookup:
            rejected_edits.append((e, "unknown segment"))
            continue
        ok, info = validate_edit(e, seg_lookup[e["segment_id"]], lexicon_terms)
        (valid_edits.append(e) if ok else rejected_edits.append((e, info)))

    # ---- apply
    refined_segments, applied_edits = [], []
    for seg in segments:
        text = seg["text"]
        seg_edits = sorted((e for e in valid_edits if e["segment_id"] == seg["id"]),
                           key=lambda x: len(x["original"]), reverse=True)
        for e in seg_edits:
            m = find_edit_match(text, e["original"])  # re-match on CURRENT text
            if m is None:
                continue
            matched = text[m["start"]:m["end"]]
            text = text[:m["start"]] + e["corrected"] + text[m["end"]:]
            applied_edits.append({**e, "matched_text": matched,
                                  "match_type": m["match_type"], "match_score": m["score"]})
        refined_segments.append({**seg, "text": text})

    # ---- invariants + diff
    raw_transcript = segments_to_text(segments)
    refined_transcript = segments_to_text(refined_segments)
    invariants = {
        "numbers_preserved": extract_numbers(raw_transcript) == extract_numbers(refined_transcript),
        "negation_modality_preserved":
            protected_tokens(raw_transcript) == protected_tokens(refined_transcript),
    }
    diff_html = difflib.HtmlDiff().make_file(
        raw_transcript.splitlines(), refined_transcript.splitlines(),
        fromdesc="RAW TRANSCRIPT", todesc="REFINED TRANSCRIPT", context=True, numlines=1)

    result = {
        "model": model_id,
        "provider": "Google Gemini API",
        "raw_transcript": raw_transcript,
        "refined_transcript": refined_transcript,
        "edits": applied_edits,
        "rejected_edits": [{"edit": e, "reason": r} for e, r in rejected_edits],
        "invariants": invariants,
        "lexicon": list(lexicon.values()),
        "diff_html": diff_html,
    }

    # ---- save (only if the caller gave a run folder)
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "raw_transcript.txt").write_text(raw_transcript, encoding="utf-8")
        (run_dir / "refined_transcript.txt").write_text(refined_transcript, encoding="utf-8")
        (run_dir / "edits.json").write_text(json.dumps(applied_edits, indent=2), encoding="utf-8")
        (run_dir / "lexicon.json").write_text(json.dumps(result["lexicon"], indent=2), encoding="utf-8")
        (run_dir / "diff.html").write_text(diff_html, encoding="utf-8")
        slim = {k: v for k, v in result.items() if k != "diff_html"}
        (run_dir / "refinement_result.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
        result["run_dir"] = str(run_dir)

    on_progress("refine", 1.0, "Refinement complete.")
    return result


# ----------------------------------------------------------------------------
# ONE-CALL ENTRY POINT FOR THE UI
# ----------------------------------------------------------------------------
def new_run_dir() -> Path:
    return DATA_DIR / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def process_audio(audio_path, api_key: Optional[str] = None,
                  on_progress: ProgressCB = _noop, run_dir: Optional[Path] = None) -> dict:
    """Audio file -> raw transcript -> refined transcript (+ edits, diff, files)."""
    resolve_api_key(api_key)  # fail fast, before the slow ASR step
    run_dir = Path(run_dir) if run_dir else new_run_dir()
    raw = transcribe_audio(audio_path, on_progress)
    return refine_transcript(raw, api_key=api_key, run_dir=run_dir, on_progress=on_progress)


# ----------------------------------------------------------------------------
# CLI (for testing without the UI)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio")
    ap.add_argument("--transcript")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    cb = lambda s, f, m: print(f"[{s} {f:4.0%}] {m}")
    try:
        if a.selftest:
            raise SystemExit(0 if run_self_test() else 1)
        if a.audio:
            out = process_audio(a.audio, on_progress=cb)
        elif a.transcript:
            out = refine_transcript(Path(a.transcript).read_text(encoding="utf-8"),
                                    run_dir=new_run_dir(), on_progress=cb)
        else:
            ap.error("give --audio, --transcript or --selftest")
        print("Saved to:", out.get("run_dir"))
    except PipelineError as e:
        raise SystemExit(f"ERROR: {e}")
