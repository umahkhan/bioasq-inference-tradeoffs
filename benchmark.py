import argparse
import gc
import os
import time
import pandas as pd
import torch
from common import MODELS, TOP_K, setup_hf_token, sync_and_gpu_memory_gb
from data import load_subset
from generate import build_prompt, load_llm
from retrieve import build_index, load_embedder, retrieve

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda", choices=["cuda", "mps"])
ap.add_argument("--models", nargs="+", default=MODELS)
ap.add_argument("--n-questions", type=int, default=20)
ap.add_argument("--n-docs", type=int, default=2000)
ap.add_argument("--out", default="results/benchmark.csv")
args = ap.parse_args()

setup_hf_token()
questions, docs = load_subset(args.n_questions, args.n_docs)
embedder = load_embedder(args.device)
index = build_index(docs, embedder)
# Retrieve once so every model sees identical passages
prompts = [build_prompt(q["question"], retrieve(q["question"], index, docs, embedder, TOP_K)) for q in questions]
del embedder, index

os.makedirs(os.path.dirname(args.out), exist_ok=True)
rows = []
for model in args.models:
    llm = load_llm(model, args.device)
    llm(prompts[0])  # untimed warm-up
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for q, prompt in zip(questions, prompts):
        start = time.perf_counter()
        answer, n_in, n_out = llm(prompt)
        seconds = time.perf_counter() - start  # llm() returns after GPU work is done (generate syncs)
        rows.append({"model": model, "question_id": q["question_id"], "answer": answer,
                     "input_tokens": n_in, "output_tokens": n_out, "seconds": seconds,
                     "tokens_per_sec": n_out / seconds, "gpu_memory_gb": sync_and_gpu_memory_gb(args.device)})
    print(f"{model}: {sum(r['tokens_per_sec'] for r in rows if r['model'] == model) / len(questions):.1f} tok/s avg")
    pd.DataFrame(rows).to_csv(args.out, index=False)  # save after each model so a later failure keeps earlier results
    del llm
    gc.collect()
    torch.cuda.empty_cache() if args.device == "cuda" else torch.mps.empty_cache()

