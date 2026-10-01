"""Milestone 4: does a small same-family draft model speed up a larger target model, and at what memory cost?
Tested at bf16 only, separate from quantization. Sweeps the draft model and the number of tokens it proposes per step (K).
Greedy speculative decoding should give the same answer as the target alone, so this is a speed/memory question."""
import argparse
import gc
import time
import mlx.core as mx
import pandas as pd
from mlx_lm import load
from mlx_gen import generate
from prompts import get_prompts

# target -> drafts. A draft must share the target's tokenizer, so it comes from the same family.
TARGETS = {"Qwen2.5-3B-Instruct": ["Qwen2.5-1.5B-Instruct", "Qwen2.5-0.5B-Instruct"],
           "Llama-3.2-3B-Instruct": ["Llama-3.2-1B-Instruct"]}
K_VALUES = [1, 2, 4, 8]  # draft tokens proposed per step

ap = argparse.ArgumentParser()
ap.add_argument("--n-questions", type=int, default=20)
ap.add_argument("--out", default="results/speculative.csv")
args = ap.parse_args()

prompts = get_prompts(args.n_questions, n_docs=2000)


def run(target, draft_name, k, model, tokenizer, draft=None):
    """Generate for every prompt; draft_name='none' (k=0) is the target-only baseline."""
    generate(model, tokenizer, prompts[0]["prompt"], draft, k)  # untimed warm-up
    mx.reset_peak_memory()  # peak now covers the resident model(s) plus generation
    rows = []
    for p in prompts:
        start = time.perf_counter()
        answer, r, accepted = generate(model, tokenizer, p["prompt"], draft, k)
        rows.append({"target": target, "draft": draft_name, "k": k, "question_id": p["question_id"], "answer": answer,
                     "output_tokens": r.generation_tokens, "seconds": time.perf_counter() - start,
                     "tokens_per_sec": r.generation_tps, "draft_token_share": accepted if draft else None,
                     "peak_memory_gb": r.peak_memory})
    return rows


rows = []
for target, draft_names in TARGETS.items():
    model, tokenizer = load(f"mlx-community/{target}-bf16")
    rows += run(target, "none", 0, model, tokenizer)
    for draft_name in draft_names:
        draft, _ = load(f"mlx-community/{draft_name}-bf16")
        for k in K_VALUES:
            rows += run(target, draft_name, k, model, tokenizer, draft)
            print(f"{target} draft={draft_name} k={k} done", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)  # save as we go
        del draft
        gc.collect()
        mx.clear_cache()
    del model
    gc.collect()
    mx.clear_cache()

df = pd.DataFrame(rows)
baseline = df[df["draft"] == "none"].set_index(["target", "question_id"])["answer"]
df["same_answer"] = [a == baseline[(t, q)] for t, q, a in zip(df["target"], df["question_id"], df["answer"])]
summary = df.groupby(["target", "draft", "k"], sort=False)[["seconds", "tokens_per_sec", "draft_token_share", "peak_memory_gb", "same_answer"]].mean()
base_speed = summary.xs("none", level="draft")["tokens_per_sec"].droplevel("k")
summary["speedup"] = summary["tokens_per_sec"] / summary.index.get_level_values("target").map(base_speed)
print(summary.round(2).to_string())
