"""Plot results/speculative.csv: per target model, how decode speed and end-to-end latency change with draft model and K."""
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})

df = pd.read_csv("results/speculative.csv")
means = df.groupby(["target", "draft", "k"])[["tokens_per_sec", "seconds", "peak_memory_gb"]].mean()

TARGETS = ["Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"]
STYLE = {"Qwen2.5-1.5B-Instruct": ("#2a78d6", "o"), "Qwen2.5-0.5B-Instruct": ("#e87ba4", "^"),  # colors match the quantization chart
         "Llama-3.2-1B-Instruct": ("#1baf7a", "s")}
K_VALUES = [1, 2, 4, 8]
ROWS = [("tokens_per_sec", "Decode tokens/sec (higher = better)"),
        ("seconds", "Seconds per question (lower = better)")]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey="row")
for col, target in enumerate(TARGETS):
    for row, (metric, ylabel) in enumerate(ROWS):
        ax = axes[row, col]
        base = means.loc[(target, "none", 0)]
        ax.axhline(base[metric], color="#52514e", linestyle="--", linewidth=1.5,
                   label=f"{target.replace('-Instruct', '')} alone ({base['peak_memory_gb']:.1f} GB)")
        for draft, (color, marker) in STYLE.items():
            if (target, draft, 1) not in means.index:
                continue
            ys = [means.loc[(target, draft, k), metric] for k in K_VALUES]
            mem = means.loc[(target, draft, 1), "peak_memory_gb"]
            ax.plot(range(4), ys, color=color, marker=marker, markersize=7, linewidth=2,
                    label=f"+ {draft.replace('-Instruct', '')} draft ({mem:.1f} GB)")
        ax.set_xticks(range(4), [str(k) for k in K_VALUES])
        ax.margins(y=0.12)
        ax.grid(axis="y", color="#e5e5e3", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if row == 0:
            ax.set_title(f"Target: {target.replace('-Instruct', '')}", fontsize=11, loc="left")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        if row == 1:
            ax.set_xlabel("K = draft tokens proposed per step")
fig.text(0.01, 0.975, "Speculative decoding: a tiny draft speeds up decoding, but not end-to-end time",
         fontsize=14, fontweight="bold", va="top")
fig.text(0.01, 0.94, "BioASQ RAG, 20 summary questions. MLX on Apple M2 Pro (32 GB), bf16, greedy decoding. Dashed line = the 3B target model alone. "
         "Seconds include prompt processing.\nLegend shows peak memory. Each point is one run of all 20 questions.",
         fontsize=9.5, color="#52514e", va="top")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig("results/speculative.png", dpi=200, facecolor="white")
