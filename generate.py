import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import MAX_NEW_TOKENS, require_device


def build_prompt(question, passages):
    context = "\n\n".join(f"[{p['id']}] {p['title']}\n{p['text']}" for p in passages)
    return ("Answer the biomedical question using only the passages below. "
            "Be concise and cite passage IDs in brackets. "
            "If the passages do not contain the answer, say so.\n\n"
            f"Passages:\n{context}\n\nQuestion: {question}")


def load_llm(model_name, device):
    """Return a function prompt -> (answer, input_tokens, output_tokens). To swap models, call this with a different name."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=require_device(device))

    def llm(prompt):
        chat = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        return tokenizer.decode(out[0, n_in:], skip_special_tokens=True), n_in, out.shape[1] - n_in

    return llm
