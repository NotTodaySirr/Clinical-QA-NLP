"""
llm_labeler.py
==============
Pseudo-labels pubmedqa_unlabeled_cleaned.csv using the Gemini Flash API
with few-shot prompting from the gold standard labeled dataset.

Pipeline
--------
1. Load few-shot examples (1 per label) from the gold standard CSV.
2. Load the cleaned unlabeled CSV.
3. Skip rows already labelled (resume support).
4. Send async batches of requests to Gemini (controlled concurrency).
5. Parse JSON responses into label_decision.
6. Append results to the output CSV every BATCH_SAVE rows.

Output columns (5-column schema)
---------------------------------
  pubid | question | context | label_decision | long_answer

Usage
-----
  python backend/AI/src/label_pipeline/llm_labeler.py

Knobs
-----
  CONCURRENCY   - parallel Gemini calls at once      (default: 30)
  BATCH_SAVE    - save to disk every N rows          (default: 500)
  MODEL         - Gemini model name
  MAX_RETRIES   - retries per row on failure         (default: 3)
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env 
load_dotenv(Path(__file__).resolve().parent / ".env")

# Configuration
MODEL        = "gemini-3.1-flash-lite"
CONCURRENCY  = 30       # parallel Gemini calls
BATCH_SAVE   = 500      # flush results to disk every N rows
MAX_RETRIES  = 3        # retries per row before marking as failed
RETRY_DELAY  = 2.0      # seconds between retries (doubles each attempt)
FEW_SHOT_PER_LABEL = 3  # number of gold examples per label classmethod

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR / ".." / ".." / "data"
INPUT_CSV    = DATA_DIR / "cleaned"  / "pubmedqa_unlabeled_cleaned.csv"
OUTPUT_CSV   = DATA_DIR / "labeled"  / "pubmedqa_pseudo_labeled.csv"
FAILED_CSV   = DATA_DIR / "labeled"  / "pubmedqa_labeling_failed.csv"
GOLD_CSV     = DATA_DIR / "raw"      / "pubmedqa_labeled_gold_standard.csv"

# Output schema
OUTPUT_COLS = ["pubid", "question", "context", "label_decision", "long_answer"]

VALID_LABELS = {"Yes", "No", "Maybe"}

# ── Few-shot example loader ───────────────────────────────────────────────────
def _load_few_shot_examples(gold_csv: Path, n_per_label: int = 1) -> str:
    """
    Loads n_per_label examples per class (Yes/No/Maybe) from the gold standard
    CSV and formats them as few-shot demonstration blocks for the prompt.
    """
    df = pd.read_csv(gold_csv)
    # Normalise label column to Title-case (gold uses lowercase: yes/no/maybe)
    df["final_decision"] = df["final_decision"].str.strip().str.capitalize()

    blocks: list[str] = []
    for label in ["Yes", "No", "Maybe"]:
        subset = df[df["final_decision"] == label].head(n_per_label)
        for _, row in subset.iterrows():
            ctx_preview = str(row["context"])[:600].strip()
            la_preview  = str(row["long_answer"])[:400].strip()
            blocks.append(
                f"Example ({label}):\n"
                f"  Question   : {str(row['question']).strip()}\n"
                f"  Context    : {ctx_preview}...\n"
                f"  Long Answer: {la_preview}...\n"
                f"  Label      : {label}\n"
            )
    return "\n".join(blocks)


# Prompt template — few-shot block injected at build time
_FEW_SHOT_HEADER = """\
You are a clinical NLP expert that classifies biomedical yes/no/maybe questions.

Below are labeled examples from a verified gold-standard dataset to guide you:

{few_shot_examples}
---
Now classify the following NEW entry using the same criteria.
"""

_TASK_BLOCK = """\
Question:
{question}

Context (supporting evidence):
{context}

Author's Answer:
{long_answer}

Task:
Based on the author's answer, classify the question as:
  "Yes"   - the answer clearly supports or confirms the question
  "No"    - the answer clearly contradicts or refutes the question
  "Maybe" - the answer is uncertain, mixed, or inconclusive

Respond ONLY with a valid JSON object in this exact format:
{{"label": "Yes" | "No" | "Maybe"}}
"""


# Gemini client
def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API")
    if not api_key:
        raise EnvironmentError("GEMINI_API environment variable is not set.\n")
    return genai.Client(api_key=api_key)


# JSON parser
def _parse_response(text: str) -> dict | None:
    """
    Extract the JSON object from the model's response.
    Handles markdown code-fences and stray surrounding text.
    Returns None if parsing fails.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find the first complete JSON object
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    label = data.get("label", "").strip().capitalize()
    if label not in VALID_LABELS:
        # Fuzzy map common variants
        for valid in VALID_LABELS:
            if valid.lower() in label.lower():
                label = valid
                break
        else:
            return None

    evidence = str(data.get("extracted_evidence", "")).strip()[:300]
    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "label_decision":     label,
        "extracted_evidence": evidence,
        "confidence":         round(confidence, 4),
    }


# Single-row labeler
async def _label_row(
    client: genai.Client,
    semaphore: asyncio.Semaphore,
    row: dict,
    prompt_header: str,        # pre-built few-shot header (shared across all rows)
) -> dict:
    """
    Call Gemini for a single row using a few-shot prompt.
    Returns a labeled row dict with a label_decision.
    """
    task_block = _TASK_BLOCK.format(
        question    = str(row.get("question", "")),
        context     = str(row.get("context",  ""))[:2000],
        long_answer = str(row.get("long_answer", "")),
    )
    prompt = prompt_header + task_block

    delay = RETRY_DELAY
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.aio.models.generate_content(
                    model   = MODEL,
                    contents= prompt,
                    config  = types.GenerateContentConfig(
                        temperature      = 0.1,   # low temperature for determinism
                        max_output_tokens= 200,
                        response_mime_type= "text/plain",
                    ),
                )
                parsed = _parse_response(response.text or "")
                if parsed:
                    return {**row, **parsed}

                # Unparseable response — retry
                print(f"  [WARN] pubid={row.get('pubid')} attempt {attempt}: "
                      f"unparseable response, retrying...")

            except Exception as exc:
                print(f"  [ERROR] pubid={row.get('pubid')} attempt {attempt}: {exc}")

            if attempt < MAX_RETRIES:
                await asyncio.sleep(delay)
                delay *= 2   # exponential backoff

    # All retries exhausted
    print(f"  [FAIL] pubid={row.get('pubid')}: marking as FAILED after {MAX_RETRIES} attempts.")
    return {
        **row,
        "label_decision":     "FAILED"
    }


# Batch saver
def _save_batch(results: list[dict], output_path: Path, failed_path: Path) -> None:
    """Append a batch of results to the output CSV, separating failed rows."""
    if not results:
        return

    ok_rows     = [r for r in results if r.get("label_decision") != "FAILED"]
    failed_rows = [r for r in results if r.get("label_decision") == "FAILED"]

    for rows, path in [(ok_rows, output_path), (failed_rows, failed_path)]:
        if not rows:
            continue
        df = pd.DataFrame(rows)[OUTPUT_COLS]
        write_header = not path.exists()
        df.to_csv(path, mode="a", index=False, header=write_header)

    print(f"  Saved {len(ok_rows)} labelled | {len(failed_rows)} failed  "
          f"→ {output_path.name}")


# Main
async def run():
    print(f"Loading cleaned data from:\n  {INPUT_CSV}\n")
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"Total rows: {total:,}")

    # Resume support: skip already-labelled pubids
    already_done: set[int] = set()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists():
        done_df = pd.read_csv(OUTPUT_CSV, usecols=["pubid"])
        already_done = set(done_df["pubid"].tolist())
        print(f"Resuming — {len(already_done):,} rows already labelled, skipping them.")

    rows_to_label = [
        row for row in df.to_dict(orient="records")
        if row["pubid"] not in already_done
    ]
    remaining = len(rows_to_label)
    print(f"Rows to label: {remaining:,}\n")

    if remaining == 0:
        print("Nothing to do — all rows are already labelled!")
        return

    client    = _get_client()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Build the shared few-shot header once (reused across all rows)
    print(f"Loading few-shot examples from:\n  {GOLD_CSV}\n")
    few_shot_examples = _load_few_shot_examples(GOLD_CSV, n_per_label=FEW_SHOT_PER_LABEL)
    prompt_header     = _FEW_SHOT_HEADER.format(few_shot_examples=few_shot_examples)
    print(f"Few-shot block ({FEW_SHOT_PER_LABEL} example/label, 3 labels = "
          f"{FEW_SHOT_PER_LABEL * 3} total examples injected into each prompt)\n")

    buffer: list[dict] = []
    start_time = time.time()
    completed  = 0

    tasks = [_label_row(client, semaphore, row, prompt_header) for row in rows_to_label]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        buffer.append(result)
        completed += 1

        # Progress
        elapsed = time.time() - start_time
        rate    = completed / elapsed if elapsed > 0 else 0
        eta     = (remaining - completed) / rate if rate > 0 else float("inf")
        print(f"\r  [{completed:>6}/{remaining}]  "
              f"{rate:.1f} rows/s  ETA: {eta/60:.1f} min  "
              f"last pubid={result.get('pubid')}",
              end="", flush=True)

        # Flush buffer to disk
        if len(buffer) >= BATCH_SAVE:
            print()
            _save_batch(buffer, OUTPUT_CSV, FAILED_CSV)
            buffer.clear()

    # Flush remaining
    if buffer:
        print()
        _save_batch(buffer, OUTPUT_CSV, FAILED_CSV)

    # Summary
    elapsed = time.time() - start_time
    print(f"\nDone! {completed:,} rows processed in {elapsed/60:.1f} min.")
    print(f"Output:  {OUTPUT_CSV}")

    if FAILED_CSV.exists():
        n_failed = len(pd.read_csv(FAILED_CSV))
        if n_failed:
            print(f"Failed:  {FAILED_CSV}  ({n_failed:,} rows — review manually)")

    # Label distribution
    result_df = pd.read_csv(OUTPUT_CSV)
    dist = result_df["label_decision"].value_counts()
    print("\nLabel distribution:")
    for label, count in dist.items():
        pct = count / len(result_df) * 100
        print(f"  {label:<8}: {count:>6,}  ({pct:.1f}%)")


if __name__ == "__main__":
    asyncio.run(run())
