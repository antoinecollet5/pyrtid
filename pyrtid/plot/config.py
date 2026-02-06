import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as font_manager

def register_default_fonts(path_to_font_files: Path) -> None:
    font_defs = (
        ("TeXGyreHeros", "TeXGyreHeros", "TeXGyreHeros", "TeXGyreHeros"),
        ("normal", "bold", "normal", "bold"),
        ("normal", "normal", "italic", "italic"),
        (
            "texgyreheros/texgyreheros-regular.otf",
            "texgyreheros/texgyreheros-bold.otf",
            "texgyreheros/texgyreheros-italic.otf",
            "texgyreheros/texgyreheros-bolditalic.otf",
        ),
    )

    for name, weight, style, path in zip(*font_defs):
        _path = path_to_font_files.joinpath(path)
        assert _path.is_file()
        font_entry = font_manager.FontEntry(
            fname=str(_path),
            name=name,
            weight=weight,
            style=style,
        )
        font_manager.fontManager.ttflist.insert(0, font_entry)


def apply_default_rc_params() -> None:
    plt.plot()
    plt.close()  # required for the plot to update
    plt.rcParams.update(
        {
            "font.sans-serif": ["TeXGyreHeros", "DejaVu Sans"],
            "font.size": 16,
            "text.usetex": False,
            "mathtext.fontset": "cm",
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
