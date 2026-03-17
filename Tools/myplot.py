import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import seaborn as sns

def plot_images_grid(images, titles=None, max_cols=5, show=True, save_as=None):
    total_images = len(images)
    cols = min(total_images, max_cols)
    rows = math.ceil(total_images / cols)

    sns.reset_orig()

    # Proporsional — 3.6 lebih pas untuk retina
    fig_width  = cols * 3.6
    fig_height = rows * 3.6

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")

        if i < total_images:
            img = images[i]
            if not isinstance(img, np.ndarray):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # tampilkan gambar
            ax.imshow(img)

            # TITLE lebih elegan
            if titles is not None and i < len(titles):
                ax.set_title(
                    titles[i],
                    fontsize=12,
                    fontweight="bold",
                    pad=6,
                    color="white",
                    backgroundcolor=(0, 0, 0, 0.55),  # transparan halus
                    loc="center"
                )

    # atur layout full & tanpa margin
    fig.tight_layout(pad=0)
    plt.subplots_adjust(
        left=0, right=1, top=1, bottom=0, 
        wspace=0.01, hspace=0.12
    )

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def bar_plot_grid(x_list, y_list, title_list=None, max_cols=2, figsize=(12, 6),
                  x_rotate=10, suptitle=None, ylim=None, axis_x=True, axis_y=True,
                  palette="Set2", title_fontsize=12, title_weight="bold"):
    
    # Set tema agar seluruh grafik lebih rapi
    sns.set_theme(style="whitegrid")

    n = len(x_list)
    rows = math.ceil(n / max_cols)
    fig, axes = plt.subplots(rows, max_cols, figsize=figsize, squeeze=False)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=15, weight="bold", y=1.03)

    for i in range(n):
        row, col = divmod(i, max_cols)
        ax = axes[row][col]

        x = x_list[i]
        y = y_list[i]

        # Warna menggunakan palet seaborn
        colors = sns.color_palette(palette, n_colors=len(x))
        bars = ax.bar(x, y, color=colors)

        if ylim is not None:
            ax.set_ylim(ylim)

        # Teks bar
        for bar in bars:
            text = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    text,
                    ha='center', va='center')

        # === TITLE LEBIH BAGUS ===
        if title_list is not None:
            ax.set_title(title_list[i],
                         fontsize=title_fontsize,
                         fontweight=title_weight,
                         pad=8)

        if not axis_x:
            ax.set_xticks([])
        if not axis_y:
            ax.set_yticks([])

        ax.tick_params(axis='x', rotation=x_rotate)

    # Hapus subplot kosong
    for j in range(n, rows * max_cols):
        row, col = divmod(j, max_cols)
        fig.delaxes(axes[row][col])

    plt.tight_layout(pad=1.2)
    plt.show()

def plot_lbp_example(img_original, img_pre, img_lbp, hist_values, show=True, save_as=None):

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Original Image
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    axes[0,0].imshow(img_rgb)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis("off")

    # Preprocessing Image
    axes[0,1].imshow(img_pre, cmap='gray')
    axes[0,1].set_title("Preprocessing Image")
    axes[0,1].axis("off")

    # LBP Image
    axes[1,0].imshow(img_lbp, cmap='gray')
    axes[1,0].set_title("LBP Image")
    axes[1,0].axis("off")

    # Histogram LBP
    axes[1,1].bar(np.arange(len(hist_values)), hist_values, width=1.0)
    axes[1,1].set_title("LBP Histogram")
    axes[1,1].set_xlabel("LBP Codes")
    axes[1,1].set_ylabel("Normalized Frequency")
    axes[1,1].set_yscale("log")

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
        
def plot_glbp_example(img_pre, img_entropy, glbp_image, hist_values, show=True, save_as=None):

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Preprocessing Image
    axes[0,0].imshow(img_pre, cmap='gray')
    axes[0,0].set_title("Preprocessing Image")
    axes[0,0].axis("off")

    # Local Entropy Map
    axes[0,1].imshow(img_entropy, cmap='gray')
    axes[0,1].set_title("Local Entropy Map (MA Candidate)")
    axes[0,1].axis("off")

    # GLBP Image
    axes[1,0].imshow(glbp_image, cmap='gray')
    axes[1,0].set_title("Graph-LBP Image")
    axes[1,0].axis("off")

    # Histogram GLBP
    axes[1,1].bar(np.arange(len(hist_values)), hist_values, width=1.0)
    axes[1,1].set_title("Graph-LBP Histogram")
    axes[1,1].set_xlabel("GLBP Codes")
    axes[1,1].set_ylabel("Normalized Frequency")
    axes[1,1].set_yscale("log")  # sesuai grafik Anda

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

# ==========================================================
# GLOBAL STYLE
# ==========================================================
sns.set_theme(style="whitegrid", context="notebook")

COLOR_TRAIN = "#1f77b4"
COLOR_VAL   = "#ff7f0e"


# ==========================================================
# HISTORY TRAINING CNN
# ==========================================================
def plot_history_cnn_grid(history_list, titles=None, show=True, save_as=None):

    metrics = [
        ("accuracy", "Accuracy"),
        ("auc", "AUC"),
        ("loss", "Loss"),
        ("precision", "Precision"),
        ("sensitivity", "Sensitivity")
    ]

    for i, h in enumerate(history_list):

        hist = h.history if hasattr(h, "history") else h
        title = titles[i] if titles and i < len(titles) else f"Model {i+1}"

        epochs = range(1, len(hist.get("loss", [])) + 1)

        fig, axes = plt.subplots(
            1, len(metrics),
            figsize=(4.5*len(metrics), 4)
        )

        fig.suptitle(title, fontsize=16, fontweight="bold")

        for ax, (key, label) in zip(axes, metrics):

            train = hist.get(key, [])
            val   = hist.get(f"val_{key}", [])

            if train:
                ax.plot(
                    epochs,
                    train,
                    marker="o",
                    linewidth=2,
                    color=COLOR_TRAIN,
                    label="Train"
                )

            if val:
                ax.plot(
                    epochs,
                    val,
                    marker="o",
                    linewidth=2,
                    linestyle="--",
                    color=COLOR_VAL,
                    label="Validation"
                )

            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.4)

            for spine in ax.spines.values():
                spine.set_alpha(0.3)

            ax.legend(fontsize=9)

        plt.tight_layout()
        
        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

# ==========================================================
# CONFUSION MATRIX
# ==========================================================
def plot_confusion_matrices_grid(conf_matrices, titles, class_names=None, main_title="", max_cols=3, show=True, save_as=None):

    n = len(conf_matrices)
    cols = min(max_cols,n)
    rows = int(np.ceil(n/cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols,4.5*rows))
    axes = np.array(axes).reshape(-1)

    for i, cm in enumerate(conf_matrices):

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[i]
        )

        axes[i].set_title(titles[i], fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i+1,len(axes)):
        axes[j].axis("off")

    fig.suptitle(main_title, fontsize=16, fontweight="bold")

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()
    
# ==========================================================
# MODEL EVALUATION METRICS
# ==========================================================
def plot_classification_metrics_bar(metrics_list, titles, main_title="Model Evaluation", show=True, save_as=None):

    metrics = ["Accuracy","Recall","Specificity","Precision","F1-Score","AUC"]

    data = np.array([
        [m.get(k,0) for k in metrics]
        for m in metrics_list
    ])

    n_models = len(titles)
    cols = 2
    rows = math.ceil(len(metrics)/cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7*cols,4.5*rows))
    axes = np.array(axes).reshape(-1)

    colors = sns.color_palette("tab10", n_models)

    for i, metric in enumerate(metrics):

        ax = axes[i]
        values = data[:,i]

        x = np.arange(len(titles))
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.6
        )

        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0,1)

        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(titles, rotation=15)

        for bar in bars:

            height = bar.get_height()

            if height <= 0:
                continue

            ax.text(
                bar.get_x()+bar.get_width()/2,
                height/2,
                f"{height:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                fontweight="bold"
            )

        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    for j in range(i+1,len(axes)):
        axes[j].axis("off")

    fig.suptitle(main_title, fontsize=16, fontweight="bold")

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

# ==========================================================
# COMPUTATIONAL TIME
# ==========================================================
def plot_computational_time(results, titles, main_title="Computational Time", show=True, save_as=None):

    times = [r.get("computational_time_sec",0) for r in results]

    plt.figure(figsize=(8,4))

    bars = plt.bar(
        titles,
        times,
        color=sns.color_palette("viridis", len(titles)),
        edgecolor="black"
    )

    for bar in bars:

        h = bar.get_height()

        plt.text(
            bar.get_x()+bar.get_width()/2,
            h,
            f"{h:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.ylabel("Time (seconds)")
    plt.title(main_title, fontsize=16, fontweight="bold")

    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


# ==========================================================
# RESOURCE USAGE
# ==========================================================
def plot_resource_usage(results, titles, main_title="Resource Usage", show=True, save_as=None):

    footprint = [r.get("memory_footprint_MB",0) for r in results]
    peak      = [r.get("memory_peak_MB",0) for r in results]
    model     = [r.get("model_size_MB",0) for r in results]

    x = np.arange(len(titles))
    w = 0.25

    plt.figure(figsize=(9,4))

    plt.bar(x-w, footprint, w, label="Footprint", color="#1f77b4")
    plt.bar(x,   peak,      w, label="Peak Memory", color="#ff7f0e")
    plt.bar(x+w, model,     w, label="Model Size", color="#2ca02c")

    plt.xticks(x, titles, rotation=20)
    plt.ylabel("Memory (MB)")

    plt.title(main_title, fontsize=16, fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


# ==========================================================
# STABILITY INDEX
# ==========================================================
def plot_stability_index(results, titles, main_title="Stability Index", show=True, save_as=None):

    plt.figure(figsize=(8,4))

    colors = sns.color_palette("tab10", len(titles))

    for r, title, c in zip(results, titles, colors):

        noise  = r.get("noise_levels", [])
        scores = r.get("scores", [])

        plt.plot(
            noise,
            scores,
            marker="o",
            linewidth=2,
            color=c,
            label=f"{title} (SI={r.get('stability_index',0):.3f})"
        )

    plt.xlabel("Noise Level")
    plt.ylabel(r.get("metric_used","Score"))

    plt.title(main_title, fontsize=16, fontweight="bold")

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()