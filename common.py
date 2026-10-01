import os

DATASET = "mattmorgis/bioasq-12b-rag"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 5
MAX_NEW_TOKENS = 384  # long answers: 256 truncated up to 20% of them
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",  # Llama is gated: accept the license on HF
]
SEED = 0


def require_device(device):
    """Fail clearly if the requested GPU is missing; never fall back to CPU."""
    import torch
    available = {"cuda": torch.cuda.is_available, "mps": torch.backends.mps.is_available}
    if not available[device]():
        raise RuntimeError(f"{device} is not available. On Kaggle, enable a GPU accelerator.")
    return device


def setup_hf_token():
    """Expose the HF token (env var or Kaggle Secret) as HF_TOKEN. Never printed."""
    if "HF_TOKEN" not in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass  # optional for ungated models


def sync_and_gpu_memory_gb(device):
    """Wait for GPU work to finish, then return GPU memory in GB (peak allocated on CUDA, driver total on MPS)."""
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e9
    torch.mps.synchronize()
    return torch.mps.driver_allocated_memory() / 1e9  # current_allocated_memory misses bf16 weights
