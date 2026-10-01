"""Plot results/quantization.csv: one panel per metric, precision on x, one line per model."""
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold"})

df = pd.read_csv("results/quantization.csv")
means = df.groupby(["model", "precision"])[["tokens_per_sec", "peak_memory_gb", "rougeL", "semantic_sim"]].mean()

PRECISIONS = ["bf16", "8bit", "4bit"]
MODELS = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order, palette validated
MARKERS = ["o", "o", "s", "s"]  # circle = Qwen, square = Llama (second cue besides color)
PANELS = [("tokens_per_sec", "Generation speed (tokens/sec, higher is better)"),
          ("peak_memory_gb", "Peak memory (GB, lower is better)"),
          ("rougeL", "Wording match: ROUGE-L vs gold answer (higher is better)"),
          ("semantic_sim", "Meaning match: embedding similarity to gold (higher is better)")]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, (metric, title) in zip(axes.flat, PANELS):
    for model, color, marker in zip(MODELS, COLORS, MARKERS):
        ax.plot(range(3), [means.loc[(model, p), metric] for p in PRECISIONS], color=color, marker=marker,
                markersize=7, linewidth=2, label=model.replace("-Instruct", ""))
    ax.set_title(title, loc="left")
    ax.set_xticks(range(3), ["bf16", "8-bit", "4-bit"])
    ax.set_xlim(-0.2, 2.2)
    ax.grid(axis="y", color="#e5e5e3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("Weight precision")
# The meaning-match scores all sit near 0.88-0.90, so the axis is zoomed; say what the scale means
unrelated = df["semantic_sim_unrelated"].mean()
axes[1, 1].set_xlabel(f"Weight precision (zoomed y-axis; an unrelated answer would score {unrelated:.2f})")

fig.text(0.01, 0.975, "Quantization: about 3x faster and 50-60% less memory at 4-bit, with a small drop in answer quality",
         fontsize=14, fontweight="bold", va="top")
fig.text(0.01, 0.94, f"BioASQ RAG, {df['question_id'].nunique()} summary questions with long gold answers. MLX on Apple M2 Pro (32 GB). "
         "Greedy decoding, max 384 new tokens.", fontsize=10, color="#52514e", va="top")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.915), ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig("results/quantization.png", dpi=200, facecolor="white")
