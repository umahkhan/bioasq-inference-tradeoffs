import json
import os
from common import TOP_K, setup_hf_token
from data import load_subset
from generate import build_prompt
from retrieve import build_index, load_embedder, retrieve


def get_prompts(n_questions, n_docs, device="mps"):
    """Retrieve top-k passages once and cache the prompts, so every model and precision sees identical input."""
    path = f"results/prompts_long_{n_questions}q_{n_docs}d.json"
    if os.path.exists(path):
        return json.load(open(path))
    setup_hf_token()
    questions, docs = load_subset(n_questions, n_docs)
    embedder = load_embedder(device)
    index = build_index(docs, embedder)
    prompts = [{"question_id": q["question_id"], "type": q["type"], "gold": q["answer"],
                "prompt": build_prompt(q["question"], retrieve(q["question"], index, docs, embedder, TOP_K))}
               for q in questions]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(prompts, open(path, "w"))
    return prompts
