from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# 5.2 the two curves side by side, the picture of the overfitting
# -> the twiny() axis is an invisible second x-axis just to align the tick labels
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    out_png = Path(__file__).resolve().parents[2] / "figures" / "losses.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # figures/ is gitignored, so it will not exist on a fresh clone
    plt.savefig(out_png, dpi=300)
    print("plot saved to", out_png)
    plt.show()
