"""Milestone 3: does quantizing a model (bf16 -> 8-bit -> 4-bit) cost answer quality on long answers, and what does it buy
in memory and speed? Quality = similarity to the BioASQ gold answer (ROUGE-L for wording, bge cosine for meaning)."""
import argparse
import gc
import time
import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import load
from rouge_score import rouge_scorer
from mlx_gen import generate
from prompts import get_prompts
from retrieve import load_embedder

MODELS = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"]
PRECISIONS = ["bf16", "8bit", "4bit"]

ap = argparse.ArgumentParser()
ap.add_argument("--n-questions", type=int, default=30)
ap.add_argument("--models", nargs="+", default=MODELS)
ap.add_argument("--out", default="results/quantization.csv")
args = ap.parse_args()

prompts = get_prompts(args.n_questions, n_docs=2000)
gold = {p["question_id"]: p["gold"] for p in prompts}

rows = []
for name in args.models:
    for precision in PRECISIONS:
        mx.reset_peak_memory()  # so peak memory covers loading the weights
        model, tokenizer = load(f"mlx-community/{name}-{precision}")
        generate(model, tokenizer, prompts[0]["prompt"])  # untimed warm-up
        for p in prompts:
            start = time.perf_counter()
            answer, r, _ = generate(model, tokenizer, p["prompt"])  # greedy, capped at MAX_NEW_TOKENS
            rows.append({"model": name, "precision": precision, "question_id": p["question_id"], "answer": answer,
                         "input_tokens": r.prompt_tokens, "output_tokens": r.generation_tokens,
                         "seconds": time.perf_counter() - start, "tokens_per_sec": r.generation_tps,
                         "peak_memory_gb": r.peak_memory})
        print(f"{name} {precision} done", flush=True)
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
        pd.DataFrame(rows).to_csv(args.out, index=False)  # save as we go

# Score every answer against its gold answer, after all MLX models are freed
df = pd.DataFrame(rows)
rouge = rouge_scorer.RougeScorer(["rougeL"])
df["gold"] = df["question_id"].map(gold)
df["rougeL"] = [rouge.score(g, a)["rougeL"].fmeasure for g, a in zip(df["gold"], df["answer"])]
embedder = load_embedder("mps")
answer_vecs = embedder.encode(df["answer"].tolist(), normalize_embeddings=True)
gold_vecs = embedder.encode(df["gold"].tolist(), normalize_embeddings=True)
df["semantic_sim"] = (answer_vecs * gold_vecs).sum(axis=1)  # cosine similarity (vectors are normalized)
df["semantic_sim_unrelated"] = (answer_vecs * np.roll(gold_vecs, 1, axis=0)).sum(axis=1)  # answer vs another question's gold: the floor
df.drop(columns="gold").to_csv(args.out, index=False)

pd.set_option("display.width", 200)
print(df.groupby(["model", "precision"], sort=False)[["tokens_per_sec", "peak_memory_gb", "rougeL", "semantic_sim"]].mean().round(2).to_string())
