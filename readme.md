```markdown
# Automated Failure Mode Clustering for Traffic Law Evaluation

This utility acts as an automated LLM judge layer. It parses the incorrect reasoning streams generated during model evaluation passes (e.g., Llama, Gemma, Qwen) and categorizes their root causes into a structured failure taxonomy.

## 📊 Error Taxonomy Categories

The framework automatically maps reasoning failures into one of the following 7 standardized categories:
* `wrong_rule` — Entirely cited or relied on an incorrect traffic regulation.
* `misapplied_rule` — Cited the correct rule but applied it incorrectly to the scenario.
* `partial_answer` — Failed to select all valid options in a multi-correct question grid.
* `visual_error` — Misinterpreted visual cues, layout markers, or traffic sign details.
* `trap_triggered` — Fell into informational or syntactic traps embedded in the question.
* `knowledge_gap` — Encountered a complete absence of the necessary domain context.
* `scenario_misread` — Misunderstood the premise, question core, or scenario state.

## 🚀 Setup & Execution

1. **Install the required runtime dependencies:**
   ```bash
   pip install -r requirements.txt

```

2. **Configure evaluation log paths:**
Open the script and update the `MODEL_CSVS` dictionary mapping to point to your target evaluation output logs:
```python
MODEL_CSVS = {
    "model_alpha": "path/to/alpha_results.csv",
    "model_beta":  "path/to/beta_results.csv",
}

```


3. **Run the pipeline:**
```bash
python failure_clustering.py

```



The script implements strict checkpointing; existing entries in the output destination file will be automatically skipped on subsequent runs. Once execution finishes, a detailed distribution metric report will be printed directly to the console.

## 🔧 Modifying the Taxonomy (Small Tweak)

The clustering pipeline relies on open-ended generation combined with strict keyword extraction to remain compatible with a wide array of local open-weights backend engines. If you wish to adapt or change the target evaluation categories, you only need to apply a **small tweak** in two places:

1. **Update the `VALID_CATEGORIES` list** at the top of the file to include your new classification tokens (ensure they are lowercase string identifiers).
2. **Expand the description block inside `prepare_prompt**` so the judge model receives clear, updated operational definitions matching your custom evaluation criteria.

```

```