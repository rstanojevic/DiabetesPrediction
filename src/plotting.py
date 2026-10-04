import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src import config


def setup_plotting():
    """Podesava globalni izgled grafikona. Poziva se jednom po skripti."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": config.FONT_SIZE,
        "axes.titlesize": config.FONT_SIZE + 2,
        "axes.labelsize": config.FONT_SIZE,
        "xtick.labelsize": config.FONT_SIZE - 1,
        "ytick.labelsize": config.FONT_SIZE - 1,
        "legend.fontsize": config.FONT_SIZE - 1,
        "figure.dpi": 100,
        "savefig.dpi": config.FIGURE_DPI,
    })


def save_figure(fig, name):
    """Snima figuru u reports/figures/ i oslobadja memoriju."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Sacuvano: {path.relative_to(config.PROJECT_ROOT)}")