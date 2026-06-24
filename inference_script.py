"""
Model: Llama-4-Scout-17B (multimodal: text + images)
Output: Simplified JSON with reasoning + A/B/C true/false
"""
import os
import gc
import csv
import json
import base64
import torch
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from datasets import load_dataset

# ==========================================
# CONFIGURATION
# ==========================================
os.environ["HUGGING_FACE_HUB_TOKEN"]  = insert_token
os.environ["VLLM_NO_USAGE_STATS"]     = "1"
os.environ["VLLM_DO_NOT_TRACK"]       = "1"

MODEL_ID      = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic"
HF_DATASET_ID = "munteanr20/romanian-driving-licence"
OUTPUT_DIR    = "./structured_output/llama4scout"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "results_structured_qwen36.csv")

BATCH_SIZE    = 8
TEST_MODE     = False


# ==========================================
# SIMPLIFIED JSON SCHEMA
# ==========================================
JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "rationament": {"type": "string"},
        "raspuns": {
            "type": "object",
            "properties": {
                "A": {"type": "boolean"},
                "B": {"type": "boolean"},
                "C": {"type": "boolean"}
            },
            "required": ["A", "B", "C"]
        }
    },
    "required": ["rationament", "raspuns"]
})


# ==========================================
# HELPERS
# ==========================================
def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def load_image(image) -> Image.Image | None:
    if image is None:
        return None
    try:
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, dict):
            if image.get("bytes"):
                return Image.open(BytesIO(image["bytes"]))
            elif image.get("path"):
                return Image.open(image["path"])
    except Exception as e:
        print(f"⚠️  Image load error: {e}")
    return None


def normalize_answer(ans: str) -> str:
    return "".join(sorted(set(c for c in str(ans).upper() if c in "ABC")))


def parse_json_answer(text: str) -> dict:
    """Parse structured simplified JSON output from model."""
    try:
        data = json.loads(text)
        raspuns = data.get("raspuns", {})
        answer = "".join(sorted(k for k, v in raspuns.items() if v is True))
        return {
            "model_answer": answer,
            "rationament":  data.get("rationament", ""),
            "raw_json":      text,
        }
    except Exception:
        return {
            "model_answer": "",
            "rationament":  "",
            "raw_json":      text,
        }


# ==========================================
# PROMPT
# ==========================================
def prepare_prompt(row) -> list:
    options = row.get("options", [])
    options_text = "\n".join(options) if isinstance(options, list) else str(options)

    system_msg = (
        "Ești un expert în legislația rutieră din România. "
        "Răspunde scurt și direct la întrebări în format JSON."
    )

    user_text = (
        f"Întrebare: {row['question']}\n\n"
        f"Opțiuni:\n{options_text}\n\n"
        "Cerință: Completează câmpul 'rationament' cu o scurtă justificare a regulii aplicate, "
        "iar în câmpul 'raspuns' marchează cu true/false opțiunile corecte (A, B, C)."
    )

    image = load_image(row.get("image"))
    if image:
        img_b64 = pil_to_base64(image)
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": user_text},
        ]
    else:
        user_content = user_text

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_content},
    ]


# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📚 Loading dataset: {HF_DATASET_ID}")
    ds = load_dataset(HF_DATASET_ID, split="train")
    df = ds.to_pandas()
    df["id"] = df["id"].astype(str)

    if TEST_MODE:
        print(f"🧪 TEST MODE: active")
    else:
        print(f"   Total: {len(df)} questions")

    # Checkpointing
    done_ids    = set()
    file_exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0
    if file_exists:
        try:
            done_df  = pd.read_csv(OUTPUT_FILE)
            done_ids = set(done_df["id"].astype(str))
            print(f"🔄 Checkpoint: skipping {len(done_ids)}")
        except:
            file_exists = False

    df_todo = df[~df["id"].isin(done_ids)].copy().reset_index(drop=True)
    print(f"📝 Remaining: {len(df_todo)}")

    if len(df_todo) == 0:
        print("✅ Done!")
        return

    gpu_count = torch.cuda.device_count()
    print(f"\n🚀 Loading {MODEL_ID} on {gpu_count} GPU(s)")

    llm = LLM(
        model=MODEL_ID,
        max_model_len=4096,
        tensor_parallel_size=gpu_count,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    )

    params = SamplingParams(
        max_tokens=400,  # Redus de la 800 deoarece output-ul este semnificativ mai scurt
        temperature=0.0,
        structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA),
    )

    params_retry = SamplingParams(
        max_tokens=400,
        temperature=0.3,
        structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA),
    )

    MAX_RETRIES = 3

    def run_single_with_retry(messages: list) -> dict:
        for attempt in range(MAX_RETRIES):
            p = params if attempt == 0 else params_retry
            try:
                responses = llm.chat([messages], p, use_tqdm=False)
                raw_json  = responses[0].outputs[0].text.strip()
                parsed    = parse_json_answer(raw_json)
                if parsed["model_answer"]:
                    return parsed
            except Exception:
                pass
        return {
            "model_answer": "UNANSWERED",
            "rationament":  "",
            "raw_json":     "",
        }

    header = [
        "id", "license_type", "category", "has_image",
        "question", "correct_answer", "model_answer", "is_correct",
        "rationament", "raw_json"
    ]

    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()

        wrong_found = 0
        target_wrong = 5 if TEST_MODE else float("inf")

        for batch_start in tqdm(range(0, len(df_todo), BATCH_SIZE), desc="Batches"):
            if wrong_found >= target_wrong:
                break

            batch          = df_todo.iloc[batch_start : batch_start + BATCH_SIZE]
            messages_batch = [prepare_prompt(row) for _, row in batch.iterrows()]

            try:
                responses = llm.chat(messages_batch, params, use_tqdm=False)
            except Exception:
                responses = [None] * len(batch)

            for j, (_, row) in enumerate(batch.iterrows()):
                if wrong_found >= target_wrong:
                    break

                try:
                    raw_json = responses[j].outputs[0].text.strip() if responses[j] else ""
                    parsed   = parse_json_answer(raw_json)
                except Exception:
                    parsed   = {"model_answer": "", "rationament": "", "raw_json": ""}

                if not parsed["model_answer"]:
                    messages = prepare_prompt(row)
                    parsed   = run_single_with_retry(messages)

                correct_norm = normalize_answer(row.get("correct_answer", ""))
                answer_norm  = normalize_answer(parsed["model_answer"])
                is_correct   = (answer_norm == correct_norm) and answer_norm != ""

                if not is_correct:
                    wrong_found += 1
                    if TEST_MODE:
                        print(f"\n❌ ID: {row['id']} | Correct: {row['correct_answer']} | Model: {parsed['model_answer']}")
                        print(f"   Rationament: {parsed['rationament'][:150]}")

                writer.writerow({
                    "id":             row["id"],
                    "license_type":   row.get("license_type", ""),
                    "category":       row.get("category", ""),
                    "has_image":      row.get("image") is not None,
                    "question":       str(row.get("question", "")),
                    "correct_answer": str(row.get("correct_answer", "")),
                    "model_answer":   parsed["model_answer"],
                    "is_correct":     is_correct,
                    "rationament":    parsed["rationament"],
                    "raw_json":       parsed["raw_json"],
                })
                f.flush()
                os.fsync(f.fileno())

    print(f"\n✅ Finished evaluation → {OUTPUT_FILE}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()