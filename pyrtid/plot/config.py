import matplotlib.pyplot as plt


def apply_default_rc_params() -> None:
    """Default parameters to obtain nice plots."""
    plt.plot()
    plt.close()  # required for the plot to update
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "DejaVu Sans"],
            "font.size": 16,
            "mathtext.fontset": "cm",
            "text.usetex": False,
            "savefig.format": "svg",
            "svg.fonttype": "none",  # to store text as text, not as path
            "savefig.facecolor": "w",
            "savefig.edgecolor": "k",
            "savefig.dpi": 300,
            "figure.constrained_layout.use": True,
            "figure.facecolor": "w",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",  # weight of the x and y labels
            "figure.titleweight": "bold",
            "axes.titlesize": 18,
            "figure.titlesize": 22,
            "animation.frame_format": "svg",
        }
    )
