from mlx_lm import stream_generate
from common import MAX_NEW_TOKENS


def generate(model, tokenizer, prompt, draft_model=None, num_draft_tokens=4):
    """Greedy generation with MLX. Returns (answer, final response stats, draft acceptance rate)."""
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    text, from_draft = "", []
    for r in stream_generate(model, tokenizer, chat, max_tokens=MAX_NEW_TOKENS,
                             draft_model=draft_model, num_draft_tokens=num_draft_tokens):
        text += r.text
        from_draft.append(r.from_draft)
    return text, r, sum(from_draft) / len(from_draft)  # share of output tokens that came from the draft model
