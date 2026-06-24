"""
Model: gpt-oss-120b (local, via vllm)
Input: simplified wrong answers from Llama 4 Scout, Gemma 3-27B, Qwen 3.6-35B
Output: CSV with failure_category + rationale per wrong answer
"""
import os
import gc
import csv
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ==========================================
# CONFIGURATION
# ==========================================
os.environ["VLLM_NO_USAGE_STATS"]     = "1"
os.environ["VLLM_DO_NOT_TRACK"]       = "1"
os.environ["VLLM_USE_V1"]             = "0"

MODEL_ID    = "openai/gpt-oss-120b"
OUTPUT_DIR  = "/export/projects/nlp/razvan.muntean/article/failure_clustering"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "failure_modes_120b.csv")

MODEL_CSVS = {
    "llama4scout": "/export/projects/nlp/razvan.muntean/article/structured_output/llama4scout/results_structured.csv",
    "gemma3":      "/export/projects/nlp/razvan.muntean/article/structured_output/gemma3_27b/results_structured.csv",
    "qwen36":      "/export/projects/nlp/razvan.muntean/article/structured_output/qwen36_35b/results_structured.csv",
}

BATCH_SIZE = 16

VALID_CATEGORIES = [
    "wrong_rule",
    "misapplied_rule",
    "partial_answer",
    "visual_error",
    "trap_triggered",
    "knowledge_gap",
    "scenario_misread",
]


# ==========================================
# PARSE — extract category from free text
# ==========================================
def extract_category(text: str) -> str:
    """Find the first valid category keyword in the model output."""
    text_lower = text.lower()
    for cat in VALID_CATEGORIES:
        if cat in text_lower:
            return cat
    return "knowledge_gap"  # genuine fallback


def extract_rationale(text: str) -> str:
    """Best-effort: grab the line after RATIONALE: or just return the full text."""
    for line in text.splitlines():
        l = line.strip()
        if l.lower().startswith("rationale:"):
            return l[len("rationale:"):].strip()
        if l.lower().startswith("reason:"):
            return l[len("reason:"):].strip()
    for line in text.splitlines():
        l = line.strip()
        if l and not any(cat in l.lower() for cat in VALID_CATEGORIES):
            return l
    return text[:200]


# ==========================================
# PROMPT (Adapted to use the single rationament field)
# ==========================================
def prepare_prompt(row, model_name: str) -> list:
    system_msg = (
        "You are an expert analyst of AI reasoning errors in Romanian traffic law exams. "
        "Classify why a model answered incorrectly. Reply with exactly two lines:\n"
        "CATEGORY: <one of the 7 categories below>\n"
        "RATIONALE: <one sentence explaining why>\n"
        "Do not add anything else."
    )

    # Fallback checking for backward compatibility with older logs
    reasoning_text = row.get("rationament", "")
    if not reasoning_text and "reguli_aplicabile" in row:
        reasoning_text = f"Rules: {row.get('reguli_aplicabile','')}. Assumptions: {row.get('presupuneri_cheie','')}. Trap: {row.get('capcana_principala','')}"

    user_text = (
        f"Model: {model_name}\n"
        f"Question: {row['question']}\n"
        f"Correct answer: {row['correct_answer']} | Model answer: {row['model_answer']}\n"
        f"Has image: {row['has_image']} | Category: {row['category']}\n\n"
        f"Model reasoning:\n"
        f"- {reasoning_text}\n\n"
        "Choose EXACTLY ONE category:\n"
        "  wrong_rule        — applied the wrong traffic rule entirely\n"
        "  misapplied_rule   — right rule, wrong application to the scenario\n"
        "  partial_answer    — missed one or more correct options (multi-answer)\n"
        "  visual_error      — misread a traffic sign, signal, or diagram\n"
        "  trap_triggered    — fell into an informational or syntactic trap\n"
        "  knowledge_gap     — lacked the specific domain knowledge\n"
        "  scenario_misread  — misunderstood the scenario or question\n"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_text},
    ]


# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    for model_name, csv_path in MODEL_CSVS.items():
        if not os.path.exists(csv_path):
            print(f"⚠️  Missing evaluation file: {csv_path}, skipping model.")
            continue
        df = pd.read_csv(csv_path)
        wrong = df[
            (df["is_correct"] == False) &
            (df["model_answer"] != "UNANSWERED")
        ].copy()
        wrong["model_name"] = model_name
        all_rows.append(wrong)
        print(f"{model_name}: {len(wrong)} wrong answers")

    if not all_rows:
        print("❌ No wrong answers files found. Check paths.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    print(f"\nTotal to classify: {len(df_all)}")

    # Checkpointing
    done_keys   = set()
    file_exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0
    if file_exists:
        try:
            done_df   = pd.read_csv(OUTPUT_FILE)
            done_keys = set(zip(done_df["id"].astype(str), done_df["model_name"]))
            print(f"🔄 Checkpoint: skipping {len(done_keys)}")
        except:
            file_exists = False

    df_todo = df_all[
        ~df_all.apply(lambda r: (str(r["id"]), r["model_name"]) in done_keys, axis=1)
    ].reset_index(drop=True)
    print(f"📝 Remaining: {len(df_todo)}")

    if len(df_todo) == 0:
        print("✅ Already done!")
        _print_summary()
        return

    gpu_count = torch.cuda.device_count()
    print(f"\n🚀 Loading {MODEL_ID} on {gpu_count} GPU(s)")

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=gpu_count,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=True,
    )

    params = SamplingParams(
        max_tokens=150,
        temperature=0.0,
    )

    header = [
        "id", "model_name", "license_type", "category", "has_image",
        "correct_answer", "model_answer",
        "failure_category", "rationale",
    ]

    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()

        for batch_start in tqdm(range(0, len(df_todo), BATCH_SIZE), desc="Classifying"):
            batch = df_todo.iloc[batch_start: batch_start + BATCH_SIZE]
            messages_batch = [
                prepare_prompt(row, row["model_name"])
                for _, row in batch.iterrows()
            ]

            try:
                responses = llm.chat(messages_batch, params, use_tqdm=False)
            except Exception as e:
                print(f"\n❌ Batch error: {e}")
                responses = [None] * len(batch)

            for j, (_, row) in enumerate(batch.iterrows()):
                try:
                    raw_text = responses[j].outputs[0].text.strip() if responses[j] else ""
                except Exception:
                    raw_text = ""

                failure_category = extract_category(raw_text)
                rationale        = extract_rationale(raw_text)

                writer.writerow({
                    "id":               row["id"],
                    "model_name":       row["model_name"],
                    "license_type":     row.get("license_type", ""),
                    "category":         row.get("category", ""),
                    "has_image":        row.get("has_image", ""),
                    "correct_answer":   row.get("correct_answer", ""),
                    "model_answer":     row.get("model_answer", ""),
                    "failure_category": failure_category,
                    "rationale":        rationale,
                })
                f.flush()

    print(f"\n✅ Done! → {OUTPUT_FILE}")
    _print_summary()

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def _print_summary():
    if not os.path.exists(OUTPUT_FILE):
        return
    df = pd.read_csv(OUTPUT_FILE)
    print("\n=== FAILURE MODE DISTRIBUTION PER MODEL ===")
    pivot = df.groupby(["model_name", "failure_category"]).size().unstack(fill_value=0)
    print(pivot.to_string())
    print("\n=== FAILURE MODE % PER MODEL ===")
    print(pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1).to_string())


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()