import argparse
import os
import pandas as pd
from common import LLM_MODEL, TOP_K, setup_hf_token
from data import load_subset
from generate import build_prompt, load_llm
from retrieve import build_index, load_embedder, retrieve

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda", choices=["cuda", "mps"])
ap.add_argument("--n-questions", type=int, default=5)
ap.add_argument("--n-docs", type=int, default=2000)
ap.add_argument("--out", default="results/rag_smoke.csv")
args = ap.parse_args()

setup_hf_token()
questions, docs = load_subset(args.n_questions, args.n_docs)
embedder = load_embedder(args.device)
index = build_index(docs, embedder)
llm = load_llm(LLM_MODEL, args.device)

rows = []
for q in questions:
    passages = retrieve(q["question"], index, docs, embedder, TOP_K)
    ids = [p["id"] for p in passages]
    hits = len(set(ids) & set(q["relevant_passage_ids"]))
    answer, _, _ = llm(build_prompt(q["question"], passages))
    rows.append({"question_id": q["question_id"], "question": q["question"], "retrieved_ids": ids,
                 "relevant_hits": hits, "n_relevant": len(q["relevant_passage_ids"]),
                 "gold_answer": q["answer"], "model": LLM_MODEL, "answer": answer})
    print(f"\nQ: {q['question']}\nhits {hits}/{len(q['relevant_passage_ids'])}\nA: {answer}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
pd.DataFrame(rows).to_csv(args.out, index=False)
