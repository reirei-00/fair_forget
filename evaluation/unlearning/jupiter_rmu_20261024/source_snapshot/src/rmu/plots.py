"""Training curves from the losses.csv that run_rmu writes next to each unlearned model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

# Colour follows the data set: blue = forget (stereotype), orange = retain (anti-stereotype).
FORGET = "#2a78d6"
RETAIN = "#eb6834"
NEUTRAL = "#52514e"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# (title, [(column, colour, label)], log scale). Each panel has its own y-axis because the
# losses differ by orders of magnitude; one shared axis would flatten the smaller ones.
PANELS = [
    ("Unlearn loss: forget activations vs control vector", [("unlearn_loss", FORGET, None)], True),
    (
        "Retain loss × α: retain activations vs original model",
        [("retain_loss", RETAIN, None)],
        False,
    ),
    (
        "Cosine similarity to the original model",
        [("unlearn_cosine_sim", FORGET, "forget"), ("retain_cosine_sim", RETAIN, "retain")],
        False,
    ),
    ("Mean gradient of the first trained matrix", [("grad_mean", NEUTRAL, None)], True),
]


def _fmt(value) -> str:
    """Readable numbers: 220,213 / 866 / 0.95 / 0.0234 (no 2.2e+05)."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def load_losses(path) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize(losses: pd.DataFrame) -> pd.DataFrame:
    """First, last and min value of each column, plus the step where NaN first appears."""
    columns = [c for c in losses.select_dtypes("number").columns if c != "step"]
    summary = pd.DataFrame(
        {
            "first": losses[columns].iloc[0],
            "last": losses[columns].iloc[-1],
            "min": losses[columns].min(),
        }
    )
    summary["first_nan_step"] = [
        losses.loc[losses[c].isna(), "step"].min() if losses[c].isna().any() else None
        for c in columns
    ]
    return summary


def _style(ax, title, log):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED, labelsize=9, length=0)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: _fmt(v)))


def _label_line_ends(ax, ends, min_gap_pt=11):
    """Write end-of-line labels, pushing them apart vertically when lines end close together
    (e.g. both cosines ending at 1)."""
    to_pt = 72 / ax.figure.dpi
    placed = []
    for x, y, text in sorted(ends, key=lambda end: end[1]):
        y_pt = ax.transData.transform((x, y))[1] * to_pt
        target = y_pt if not placed else max(y_pt, placed[-1] + min_gap_pt)
        placed.append(target)
        ax.annotate(
            text,
            (x, y),
            xytext=(6, target - y_pt),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=MUTED,
        )


def plot_losses(losses: pd.DataFrame, title: str = "", save_to=None):
    """One panel per quantity, sharing the step axis. Panels whose columns are missing
    (e.g. cosines without --verbose) are skipped. A dashed line marks the first NaN step."""
    panels = [
        (panel_title, [s for s in series if s[0] in losses], log)
        for panel_title, series, log in PANELS
    ]
    panels = [p for p in panels if p[1]]

    fig, axes = plt.subplots(
        len(panels), 1, sharex=True, figsize=(9, 2.3 * len(panels)), facecolor=SURFACE
    )
    axes = [axes] if len(panels) == 1 else list(axes)
    steps = losses["step"]

    for ax, (panel_title, series, log) in zip(axes, panels):
        values = losses[[s[0] for s in series]]
        valid = values.stack()
        # Log only when values are positive and span more than ~1.5 decades; otherwise a
        # log axis shows a single tick.
        wide = bool((valid > 0).all()) and valid.max() / valid.min() > 30 if len(valid) else False
        _style(ax, panel_title, log and wide)
        ends = []
        for column, colour, label in series:
            ax.plot(steps, losses[column], color=colour, linewidth=2, label=label)
            last = losses[column].last_valid_index()
            if last is not None:
                # Direct label at the end of the line: final value (and series name).
                text = _fmt(losses[column][last])
                if label:
                    text = f"{label} {text}"
                ends.append((steps[last], losses[column][last], text))
        _label_line_ends(ax, ends)
        if any(label for _, _, label in series):
            ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="best")

        nan_steps = steps[values.isna().any(axis=1)]
        if not nan_steps.empty:
            ax.axvline(nan_steps.min(), color=MUTED, linewidth=1, linestyle="--")
        if not nan_steps.empty and ax is axes[0]:
            ax.text(
                nan_steps.min(),
                1,
                f"NaN from step {nan_steps.min()} ",
                transform=ax.get_xaxis_transform(),
                va="top",
                ha="right",
                fontsize=9,
                color=MUTED,
            )

    axes[-1].set_xlabel("Training step", color=MUTED, fontsize=9)
    axes[-1].margins(x=0.08)  # room for the end-of-line labels
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    if save_to:
        fig.savefig(Path(save_to), dpi=150, facecolor=SURFACE)
    return fig


# Layer sweep: the two terms of the forget-loss drop. Violet = rotation towards u (the
# intended misdirection), yellow = shrinkage. Retain bars use the same colours, lighter.
SHRINKAGE = "#eda100"
ROTATION = "#6250d6"


def plot_layer_sweep(sweep: pd.DataFrame, title: str = "", retain: bool = True, save_to=None):
    """Stacked bar per layer from sweep.csv: shrinkage (1 - k^2)/2 at the bottom, rotation
    k*cos(theta) on top, so the bar height approximates the relative drop 1 - L/L0. A dot
    marks the measured drop. With ``retain`` a thinner, lighter bar shows the same terms
    on retain data next to each forget bar.
    """
    sweep = sweep.sort_values("layer").reset_index(drop=True)
    x = np.arange(len(sweep))
    width = 0.38 if retain else 0.6
    forget_x = x - width / 2 if retain else x

    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(sweep) + 2), 4.2), facecolor=SURFACE)
    _style(ax, "Relative drop of the forget loss, 1 − L / L₀", log=False)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))

    def bars(xs, prefix, w, alpha, label_suffix):
        # Surface-coloured edges give the 2px gap between stacked segments and neighbours.
        style = dict(width=w, edgecolor=SURFACE, linewidth=1.5, alpha=alpha)
        ax.bar(
            xs,
            sweep[f"{prefix}_shrinkage"],
            color=SHRINKAGE,
            label=f"shrinkage{label_suffix}",
            **style,
        )
        ax.bar(
            xs,
            sweep[f"{prefix}_rotation"],
            bottom=sweep[f"{prefix}_shrinkage"],
            color=ROTATION,
            label=f"rotation{label_suffix}",
            **style,
        )

    bars(forget_x, "forget", width, 1.0, " (forget)" if retain else "")
    if retain:
        bars(x + width / 2, "retain", width * 0.6, 0.4, " (retain)")
    ax.scatter(
        forget_x,
        sweep["forget_drop"],
        s=36,
        color=INK,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=3,
        label="measured drop (forget)",
    )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, [str(layer) for layer in sweep["layer"]])
    ax.set_xlabel("Hooked layer (trained: layer − 2 … layer)", color=MUTED, fontsize=9)
    # Below the axis: real bars can reach any height, so no corner is safe.
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=MUTED,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    if save_to:
        fig.savefig(Path(save_to), dpi=150, facecolor=SURFACE)
    return fig
