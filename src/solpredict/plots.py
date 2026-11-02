from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd 

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a2"
NAMES = {"persistence": "Tomorrow = today", "seasonal": "Typical day for the season",
         "ridge": "Linear (ridge)", "boosting": "Gradient boosting"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _axes(w, h, n=1, **kw):
    fig, axes = plt.subplots(1, n, figsize=(w, h), dpi=150, **kw)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
    return fig, axes


def skill_bars(test: dict, path) -> None:
    order = ["persistence", "seasonal", "ridge", "boosting"]
    rmse = [test[k]["rmse_wm2"] for k in order]
    fig, ax = _axes(6.6, 3.2)
    colors = [GREY, GREY, BLUE, BLUE]
    y = np.arange(len(order))
    ax.barh(y, rmse, color=colors, height=0.6, zorder=2)
    for yy, k, r in zip(y, order, rmse):
        sk = test[k]["skill_vs_persistence"]
        label = f"{r:.0f} W/m²" + ("" if k == "persistence" else f"  ({sk:+.0%} vs tomorrow = today)")
        ax.text(r + max(rmse) * 0.01, yy, label, va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [NAMES[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlim(0, max(rmse) * 1.8)
    ax.set_xlabel("Hourly error on test days 2024–2025 (RMSE, daylight hours, W/m²)", color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_title("Lower is better. Grey: no-model baselines", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def by_month(table: pd.DataFrame, model: str, path) -> None:
    fig, ax = _axes(6.6, 3.2)
    x = table.index.to_numpy()
    ax.plot(x, table["persistence_rmse"], color=GREY, linewidth=2, marker="o", markersize=4,
            label=NAMES["persistence"])
    ax.plot(x, table["model_rmse"], color=BLUE, linewidth=2, marker="o", markersize=4, label=NAMES[model])
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_ylabel("RMSE, W/m²", color=INK_2)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title("Hourly error by month, test days 2024–2025", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def example_days(ex, forecast, persistence, worst: pd.DataFrame, test_mask, model: str, path) -> None:
    sub = ex[test_mask].assign(f=np.asarray(forecast)[test_mask.to_numpy()],
                               p=np.asarray(persistence)[test_mask.to_numpy()])
    daily = sub[sub["daylight"]].groupby("date").apply(lambda g: abs(g["f"].sum() - g["ghi"].sum()),
                                                        include_groups=False)
    typical = daily.sort_values().index[len(daily) // 2]
    worst_day = worst.index[0]
    fig, axes = _axes(9, 3.2, n=2, sharey=True)
    for ax, d, title in ((axes[0], typical, "Typical day"), (axes[1], worst_day, "Worst day")):
        g = sub[sub["date"] == d]
        ax.plot(g["hour"], g["ghi"], color=INK, linewidth=2, label="Actual")
        ax.plot(g["hour"], g["f"], color=BLUE, linewidth=2, label=NAMES[model])
        ax.plot(g["hour"], g["p"], color=GREY, linewidth=1.5, linestyle="--", label=NAMES["persistence"])
        cause = f", likely {worst.loc[d, 'likely_cause']}" if d in worst.index else ""
        ax.set_title(f"{title}: {pd.Timestamp(d):%d %b %Y}{cause}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Hour (local solar time)", color=INK_2)
        ax.set_xlim(4, 20)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    axes[0].set_ylabel("Sunlight at ground (GHI), W/m²", color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def data_weeks(df: pd.DataFrame, path, summer: str = "2023-06-10", winter: str = "2023-01-10") -> None:
    fig, axes = _axes(10, 3.4, n=2, sharey=True)
    for ax, start, title in ((axes[0], winter, "Winter week"), (axes[1], summer, "Summer week")):
        w = df.loc[start:pd.Timestamp(start) + pd.Timedelta(days=6, hours=23)]
        ax.plot(w.index, w["TOA_SW_DWN"], color=GREY, linewidth=1.2, label="Top of atmosphere")
        ax.plot(w.index, w["CLRSKY_SFC_SW_DWN"], color=ORANGE, linewidth=1.2, label="Clear sky (no clouds)")
        ax.plot(w.index, w["ALLSKY_SFC_SW_DWN"], color=BLUE, linewidth=1.6, label="Actual at ground (GHI)")
        ax.set_title(f"{title} from {pd.Timestamp(start):%d %b %Y}", color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("W/m²", color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def best_worst(sub: pd.DataFrame, best, worst, path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for row, days, label in ((0, best, "Best"), (1, worst, "Worst")):
        for col, d in enumerate(days):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            g = sub[sub["date"] == d]
            ax.plot(g["hour"], g["ghi"], color=INK, linewidth=2, label="Actual")
            ax.plot(g["hour"], g["f"], color=BLUE, linewidth=2, label="Forecast")
            ax.set_title(f"{label} {col + 1}: {pd.Timestamp(d):%d %b %Y}", color=INK, fontsize=9, loc="left")
            ax.set_xlim(5, 19)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.tick_params(colors=INK_2, labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    for r in range(2):
        axes[r, 0].set_ylabel("GHI, W/m²", color=INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
